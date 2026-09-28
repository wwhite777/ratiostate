#!/usr/bin/env python3
"""Private saved-record conditional-score replay; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORER_PATH = HERE / 'score_lineage.py'
if hashlib.sha256(SCORER_PATH.read_bytes()).hexdigest() != 'c6d6543792190cdee25fb33ef570e9185d7fd5d8735561fc905c7b1c9e0a6cdc':
    raise ImportError('adjacent frozen scorer hash mismatch')
SCORER_SPEC = importlib.util.spec_from_file_location('ratiostate_score_lineage', SCORER_PATH)
if SCORER_SPEC is None or SCORER_SPEC.loader is None:
    raise ImportError('unable to load adjacent scorer')
SCORER_MODULE = importlib.util.module_from_spec(SCORER_SPEC)
SCORER_SPEC.loader.exec_module(SCORER_MODULE)
score = SCORER_MODULE.score
ARMS = ('A00', 'A10', 'A01', 'A11', 'fp16_boundary', 'bf16_pair_boundary')
CONTRASTS = {
    'normalizer_given_bf16_matrix': (('A11', 1), ('A10', -1)),
    'matrix_only': (('A10', 1), ('A00', -1)),
    'normalizer_only': (('A01', 1), ('A00', -1)),
    'both': (('A11', 1), ('A00', -1)),
    'interaction': (('A11', 1), ('A10', -1), ('A01', -1), ('A00', 1)),
    'matrix_given_bf16_normalizer': (('A11', 1), ('A01', -1)),
    'A10_minus_fp16': (('A10', 1), ('fp16_boundary', -1)),
    'fp16_minus_native': (('fp16_boundary', 1), ('A00', -1)),
    'pair_minus_native': (('bf16_pair_boundary', 1), ('A00', -1)),
}
SEED, DRAWS = 20260924, 20000

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def within(relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('bundle member path must be relative')
    path = (HERE / relative).resolve()
    if not path.is_relative_to(HERE) or not path.is_file():
        raise ValueError(f'bundle member missing or escapes bundle: {relative}')
    return path

def verify_package() -> str:
    manifest = json.loads((HERE / 'package_manifest.json').read_text())
    if manifest.get('status') != 'private_candidate' or manifest.get('schema') != 'RatioState-fixed32-saved-record-v1':
        raise ValueError('package manifest scope mismatch')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('empty package manifest')
    if any('__pycache__' in Path(name).parts or Path(name).suffix in ('.pyc', '.pyo') for name in files):
        raise ValueError('bytecode must not be packaged')
    if any('__pycache__' in path.relative_to(HERE).parts or path.suffix in ('.pyc', '.pyo')
           for path in HERE.rglob('*') if path.is_file()):
        raise ValueError('bytecode present in portable bundle')
    for relative, record in files.items():
        path = within(relative)
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'package input hash mismatch: {relative}')
    return digest(HERE / 'package_manifest.json')

def replay_scores(rows: list[dict]) -> list[dict]:
    results = []
    seen = set()
    for row_number, row in enumerate(rows):
        cohort, index, label = row['cohort'], row['index'], row['label']
        if cohort not in ('development10', 'fixed32') or type(index) is not int or type(label) is not int:
            raise ValueError('invalid cohort/index/label')
        if cohort != ('development10' if row_number < 10 else 'fixed32'):
            raise ValueError('cohort order mismatch')
        if (cohort, index) in seen:
            raise ValueError('duplicate cohort/index')
        seen.add((cohort, index))
        if cohort == 'fixed32' and row.get('ordinal') != row_number - 10:
            raise ValueError('fixed32 ordinal mismatch')
        pixels = json.loads(within(row['pixels']).read_text())
        if pixels.get('index') != index or pixels.get('label') != label or pixels.get('image_shape') != [28, 28]:
            raise ValueError(f'pixel identity mismatch: {cohort} {index}')
        values = pixels.get('pixels')
        if not isinstance(values, list) or len(values) != 784 or any(type(x) is not int or not 0 <= x <= 255 for x in values):
            raise ValueError(f'invalid pixels: {cohort} {index}')
        targets = np.asarray(values[1:], dtype=np.int64)
        score_records = row['scores']
        missing = row.get('missingReasons', {})
        if not isinstance(score_records, dict) or not isinstance(missing, dict):
            raise ValueError('scores and missing reasons must be objects')
        if set(score_records) - set(ARMS) or set(missing) - set(ARMS):
            raise ValueError('unknown arm')
        if set(score_records) & set(missing):
            raise ValueError(f'present score also marked missing: {cohort} {index}')
        if any(not isinstance(reason, str) or not reason for reason in missing.values()):
            raise ValueError(f'empty missing reason: {cohort} {index}')
        if set(score_records) | set(missing) != set(ARMS):
            raise ValueError(f'arm coverage or missing reason absent: {cohort} {index}')
        computed = {}
        for arm in ARMS:
            if arm not in score_records:
                continue
            source = score_records[arm]
            prediction = within(source['prediction'])
            if prediction.stat().st_size != 783 * 30 * 4:
                raise ValueError(f'prediction byte length mismatch: {cohort} {index} {arm}')
            params = np.fromfile(prediction, dtype='<f4').reshape(783, 30)
            if not np.isfinite(params).all():
                raise ValueError(f'nonfinite prediction: {cohort} {index} {arm}')
            token = score(params, targets)['token_log_probability']
            total_nll = float(-np.sum(token))
            bits = total_nll / (math.log(2.0) * 783)
            expected = float(source['expectedBits'])
            if not math.isclose(bits, expected, rel_tol=2e-13, abs_tol=2e-13):
                raise ValueError(f'expected score mismatch: {cohort} {index} {arm}')
            computed[arm] = bits
            results.append({'cohort': cohort, 'index': index, 'label': label, 'arm': arm, 'bits': bits,
                            'savedBits': source['expectedBits'], 'predictionSha256': digest(prediction)})
        row['_computed'] = computed
    if sum(row['cohort'] == 'development10' for row in rows) != 10 or sum(row['cohort'] == 'fixed32' for row in rows) != 32:
        raise ValueError('expected ten development and 32 fixed images')
    if len(results) == 0:
        raise ValueError('empty score replay')
    return results

def contrast_results(rows: list[dict]) -> tuple[list[dict], dict]:
    per_image, summary = [], {}
    for cohort in ('development10', 'fixed32'):
        cohort_rows = [row for row in rows if row['cohort'] == cohort]
        summary[cohort] = {}
        for name, terms in CONTRASTS.items():
            values = []
            decimal_disagreements = 0
            max_discrepancy = 0.0
            for row in cohort_rows:
                scores = row['_computed']
                if not all(arm in scores for arm, _ in terms):
                    continue
                value = float(sum(factor * scores[arm] for arm, factor in terms))
                with localcontext() as context:
                    context.prec = 80
                    precise = sum(Decimal(str(row['scores'][arm]['expectedBits'])) * factor for arm, factor in terms)
                delta = abs(value - float(precise))
                max_discrepancy = max(max_discrepancy, delta)
                if (value > 0) - (value < 0) != (precise > 0) - (precise < 0):
                    decimal_disagreements += 1
                per_image.append({'cohort': cohort, 'index': row['index'], 'label': row['label'],
                                  'contrast': name, 'value': value, 'decimalValue': str(precise)})
                values.append(value)
            values_array = np.asarray(values, dtype=np.float64)
            item = {'n': len(values), 'missing': len(cohort_rows) - len(values),
                    'decimalSignDisagreements': decimal_disagreements, 'maximumDecimalDiscrepancy': max_discrepancy,
                    'bootstrap': None}
            if values:
                item.update(mean=float(np.mean(values_array)), median=float(np.median(values_array)),
                            minimum=float(np.min(values_array)), maximum=float(np.max(values_array)),
                            signs={'positive': int(np.sum(values_array > 0)), 'negative': int(np.sum(values_array < 0)), 'zero': int(np.sum(values_array == 0))})
            if len(values) >= 2:
                item['sampleSD'] = float(np.std(values_array, ddof=1))
                if cohort == 'fixed32':
                    rng = np.random.Generator(np.random.PCG64(SEED))
                    draws = rng.integers(0, len(values), size=(DRAWS, len(values)))
                    means = values_array[draws].mean(axis=1, dtype=np.float64)
                    item['bootstrap'] = {'lower': float(np.quantile(means, 0.025, method='linear')),
                                         'upper': float(np.quantile(means, 0.975, method='linear')),
                                         'draws': DRAWS, 'seed': SEED, 'bitGenerator': 'PCG64', 'unit': 'complete paired image'}
            summary[cohort][name] = item
    return per_image, summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(HERE):
        print('refusing existing or in-bundle output directory', file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    try:
        if np.__version__ != '1.26.4':
            raise ValueError('NumPy version mismatch')
        if os.getenv('CUDA_VISIBLE_DEVICES') != '' or any(os.getenv(k) != '2' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
            raise ValueError('CPU/numerical-thread environment mismatch')
        before = verify_package()
        inputs = json.loads((HERE / 'inputs/rows.json').read_text())
        if inputs.get('metric') != 'idealized_sampler_implied_conditional_discretized_nll':
            raise ValueError('metric mismatch')
        rows = inputs['rows']
        scores = replay_scores(rows)
        per_image, summary = contrast_results(rows)
        if verify_package() != before:
            raise ValueError('package changed during replay')
        write_json(output / 'scores.json', {'rows': scores, 'metric': inputs['metric'], 'conditioning': 'pixel 0; score pixels 1–783'})
        write_json(output / 'contrasts.json', {'perImage': per_image, 'summary': summary,
            'fixed32Bootstrap': {'draws': DRAWS, 'seed': SEED, 'bitGenerator': 'PCG64', 'quantileMethod': 'linear', 'pointwise': True}})
        write_json(output / 'provenance.json', {'packageManifestSha256': before, 'numpy': np.__version__,
            'scorerSha256': digest(HERE / 'score_lineage.py'), 'scope': 'saved-record replay only; no model inference'})
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in output.iterdir() if p.is_file()}
        write_json(output / 'manifest.json', {'files': files})
        write_json(output / 'complete.json', {'status': 'complete', 'manifestSha256': digest(output / 'manifest.json')})
        print(json.dumps({'status': 'complete', 'scores': len(scores), 'images': len(rows), 'output': str(output)}))
        return 0
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'errorType': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
