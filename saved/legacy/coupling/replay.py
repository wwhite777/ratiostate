#!/usr/bin/env python3
"""Portable replay of the saved Section 5.4 fixed-feature decomposition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "COUPLING_DIAGNOSTIC_CONTRACT_v1.md"
ARMS = ("fp32", "bf16", "bf16_scale_half")
LAYERS = (0, 7)
SHAPE_VECTOR = (783, 1, 2, 1, 32)
SHAPE_SCALAR = (783, 1, 2, 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_package_manifest() -> str:
    path = ROOT / "package_manifest.json"
    document = json.loads(path.read_text())
    require(document.get("status") == "private_candidate" and document.get("scope") == "authored-derived fixed-feature Section 5.4 inputs only", "invalid package scope")
    files = document.get("files")
    require(isinstance(files, dict) and files, "missing package file manifest")
    for name, record in files.items():
        filename = (ROOT / name).resolve()
        require(filename.is_relative_to(ROOT) and filename.is_file(), f"invalid package path: {name}")
        require(filename.stat().st_size == record["bytes"] and sha(filename) == record["sha256"], f"package input hash mismatch: {name}")
    return sha(path)


def compare_expected_rows(rows: list[dict]) -> None:
    expected = json.loads((ROOT / "expected_rows.json").read_text())["rows"]
    require(len(expected) == len(rows) == 24, "expected row count mismatch")
    def equal(a, b, label):
        if isinstance(a, dict) and isinstance(b, dict):
            require(a.keys() == b.keys(), f"expected row keys mismatch: {label}")
            for key in a:
                equal(a[key], b[key], f"{label}.{key}")
        elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            require(a == b and type(a) is type(b), f"expected row flag mismatch: {label}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            require(math.isclose(a, b, abs_tol=2e-13, rel_tol=2e-13), f"expected numeric row mismatch: {label}")
        else:
            require(a == b, f"expected row mismatch: {label}")
    for index, (actual, old) in enumerate(zip(rows, expected)):
        equal(actual, old, f"row[{index}]")


def verify_manifest(folder: Path) -> dict:
    manifest_path = folder / "manifest.json"
    complete_path = folder / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    require(complete.get("status") == "complete" and complete.get("manifestSha256") == sha(manifest_path), f"completion mismatch: {folder}")
    files = manifest.get("files")
    require(isinstance(files, dict), f"missing file manifest: {folder}")
    for name, record in files.items():
        filename = (folder / name).resolve()
        require(filename.is_relative_to(folder.resolve()) and filename.is_file(), f"invalid manifest path: {filename}")
        require(filename.stat().st_size == record["bytes"] and sha(filename) == record["sha256"], f"file hash mismatch: {filename}")
    return {"folder": str(folder.relative_to(ROOT)), "manifestSha256": sha(manifest_path), "completeSha256": sha(complete_path), "files": {name: record["sha256"] for name, record in files.items()}}


def load_trace(folder: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((folder / "trace.json").read_text())
    require(metadata["selection"] == {"layers": [0, 7], "heads": [0]}, "unexpected trace selection")
    require(metadata["input"]["steps"] == 783 and metadata["input"]["startIndex"] == 0, "unexpected trace indices")
    require(metadata["operator"]["updateOrder"].startswith("inclusive"), "unexpected update order")
    epsilons = metadata["operator"]["epsilonByLayer"]
    require(len(epsilons) == 2 and all(math.isfinite(x) and x > 0 for x in epsilons), "invalid additive epsilon")
    arrays = {}
    for name in ("q", "k", "v"):
        spec = metadata["arrays"][name]
        require(spec["shape"] == list(SHAPE_VECTOR) and spec["dtype"] == "float32" and spec["byteOrder"] == "little-endian", f"trace schema mismatch: {name}")
        values = np.fromfile(folder / spec["file"], dtype="<f4")
        require(values.size == math.prod(SHAPE_VECTOR), f"trace size mismatch: {name}")
        arrays[name] = values.reshape(SHAPE_VECTOR)
        require(np.isfinite(arrays[name]).all(), f"nonfinite mapped {name}")
    require(np.all(arrays["q"] >= 0) and np.all(arrays["k"] >= 0), "negative mapped feature")
    return metadata, arrays


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, k, v = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    state_kv = np.zeros((32, 32), dtype=np.float64)
    state_kc = np.zeros(32, dtype=np.float64)
    numerator = np.empty((783, 32), dtype=np.float64)
    denominator = np.empty(783, dtype=np.float64)
    for t in range(783):
        state_kv += np.outer(k[t], v[t])
        state_kc += k[t]
        numerator[t] = q[t] @ state_kv
        denominator[t] = q[t] @ state_kc + epsilon
    require(np.isfinite(numerator).all() and np.isfinite(denominator).all() and np.all(denominator > 0), "invalid FP64 reference")
    output = numerator / denominator[:, None]
    require(np.isfinite(output).all(), "nonfinite FP64 reference output")
    return numerator, denominator, output


def decompose(n: np.ndarray, d: np.ndarray, y: np.ndarray, saved_n: np.ndarray, saved_d: np.ndarray, saved_y: np.ndarray, scale: float) -> dict[str, np.ndarray | float | None | bool]:
    require(scale in (1.0, 0.5), "invalid representation scale")
    n = np.asarray(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nhat = np.asarray(saved_n, dtype=np.float64) / scale
    dhat = np.asarray(saved_d, dtype=np.float64) / scale
    yhat = np.asarray(saved_y, dtype=np.float64)
    require(n.shape == nhat.shape == y.shape == yhat.shape and d.shape == dhat.shape == n.shape[:-1], "decomposition shape mismatch")
    require(all(np.isfinite(x).all() for x in (n, d, y, nhat, dhat, yhat)), "nonfinite decomposition input")
    require(np.all(d > 0) and np.all(dhat > 0), "nonpositive denominator")
    a = (nhat - n) / dhat[:, None]
    b = -y * ((dhat - d) / dhat)[:, None]
    c = yhat - nhat / dhat[:, None]
    e = yhat - y
    coupled = a + b
    residual = e - (coupled + c)
    bound = 1e-12 * max(1.0, float(np.max(np.abs(e))), float(np.max(np.abs(y))), float(np.max(np.abs(yhat))))
    require(float(np.max(np.abs(residual))) <= bound, "coupled identity exceeded frozen tolerance")
    sum_a = float(np.sum(np.abs(a), dtype=np.float64))
    sum_b = float(np.sum(np.abs(b), dtype=np.float64))
    sum_coupled = float(np.sum(np.abs(coupled), dtype=np.float64))
    contribution_sum = sum_a + sum_b
    cancellation = None if contribution_sum == 0 else 1.0 - sum_coupled / contribution_sum
    if cancellation is not None:
        require(-1e-12 <= cancellation <= 1.0 + 1e-12, "cancellation outside frozen endpoint tolerance")
    return {"a": a, "b": b, "c": c, "e": e, "a_plus_b": coupled, "identity_residual": residual, "Nhat": nhat, "Dhat": dhat, "sumAbsA": sum_a, "sumAbsB": sum_b, "sumAbsAplusB": sum_coupled, "cancellationC1": cancellation, "zeroContributionSum": contribution_sum == 0, "identityBound": bound}


def fixture_results() -> dict:
    def case(n, d, saved_n, saved_d, saved_y=None, scale=1.0):
        n = np.asarray(n, dtype=np.float64).reshape(1, -1)
        d = np.asarray([d], dtype=np.float64)
        saved_n = np.asarray(saved_n, dtype=np.float64).reshape(1, -1)
        saved_d = np.asarray([saved_d], dtype=np.float64)
        y = n / d[:, None] if d[0] > 0 else np.zeros_like(n)
        yhat = (saved_n / saved_d[:, None] if saved_d[0] > 0 else np.zeros_like(saved_n)) if saved_y is None else np.asarray(saved_y, dtype=np.float64).reshape(1, -1)
        return decompose(n, d, y, saved_n, saved_d, yhat, scale)
    checks = {}
    for label, new_n, new_d in (("proportional_positive", 3.0, 6.0), ("proportional_negative", 1.0, 2.0)):
        got = case([2.0], 4.0, [new_n], new_d)
        require(got["cancellationC1"] == 1.0 and np.all(got["a_plus_b"] == 0), f"{label} failed")
        checks[label] = {"cancellationC1": got["cancellationC1"], "a": float(got["a"][0, 0]), "b": float(got["b"][0, 0])}
    got = case([2.0], 4.0, [3.0], 4.0)
    require(got["cancellationC1"] == 0.0 and np.all(got["b"] == 0), "numerator-only fixture failed")
    checks["numerator_only"] = {"cancellationC1": got["cancellationC1"]}
    got = case([2.0], 4.0, [3.0], 2.0)
    require(got["cancellationC1"] == 0.0 and np.all(got["a"] > 0) and np.all(got["b"] > 0), "reinforcing fixture failed")
    checks["reinforcing"] = {"cancellationC1": got["cancellationC1"]}
    got = case([2.0, 0.0], 4.0, [2.0, 1.0], 2.0)
    require(got["cancellationC1"] == 0.0 and float(np.sum(got["a"] * got["b"])) == 0.0, "orthogonal fixture failed")
    checks["orthogonal"] = {"cancellationC1": got["cancellationC1"], "l2TriangleSlack": float(1.0 - np.linalg.norm(got["a_plus_b"]) / (np.linalg.norm(got["a"]) + np.linalg.norm(got["b"])))}
    got = case([2.0], 4.0, [2.0], 4.0)
    require(got["cancellationC1"] is None and got["zeroContributionSum"], "zero contribution fixture failed")
    checks["zero_contribution"] = {"cancellationC1": None, "zeroContributionSum": True}
    got = case([2.0], 4.0, [2.0], 4.0, [0.625])
    require(got["c"][0, 0] == 0.125 and got["e"][0, 0] == 0.125, "output-rounding fixture failed")
    checks["output_rounding"] = {"c": float(got["c"][0, 0])}
    for label, d, saved_d in (("zero_reference_denominator", 0.0, 4.0), ("invalid_saved_denominator", 4.0, 0.0), ("negative_saved_denominator", 4.0, -1.0)):
        try:
            case([2.0], d, [2.0], saved_d)
        except ValueError:
            checks[label] = {"rejected": True}
        else:
            raise AssertionError(f"{label} did not fail")
    correct = case([2.0], 4.0, [1.5], 2.0, [0.75], scale=0.5)
    wrong = case([2.0], 4.0, [1.5], 2.0, [0.75], scale=1.0)
    require(correct["cancellationC1"] == 0.0 and wrong["cancellationC1"] > 0.5, "half-scale normalization fixture failed")
    checks["half_scale"] = {"correctC1": correct["cancellationC1"], "omittedNormalizationC1Rejected": wrong["cancellationC1"]}
    return {"status": "pass", "checks": checks}


def row_summary(image: int, layer: int, arm: str, reference_match: float, d: np.ndarray, y: np.ndarray, got: dict, old_metric: float) -> dict:
    reference_norm = float(np.linalg.norm(y))
    values = {name: np.asarray(got[name]) for name in ("a", "b", "a_plus_b", "c", "e")}
    raw_norms = {name: float(np.linalg.norm(array)) for name, array in values.items()}
    rel_norms = {name: (norm / reference_norm if reference_norm != 0 else None) for name, norm in raw_norms.items()}
    if rel_norms["e"] is not None:
        require(math.isclose(rel_norms["e"], old_metric, abs_tol=2e-13, rel_tol=2e-13), "old control metric mismatch")
    return {"image": image, "layer": layer, "head": 0, "arm": arm, "positions": 783, "coordinatesPerPosition": 32, "scalarCount": 783 * 32, "scaleUndone": 0.5 if arm == "bf16_scale_half" else 1.0, "referenceNormL2": reference_norm, "referenceNormZero": reference_norm == 0, "relativeNormUndefinedReason": "zero_reference_norm" if reference_norm == 0 else None, "rawNormL2": raw_norms, "relativeNormL2": rel_norms, "maxAbsIdentityResidual": float(np.max(np.abs(got["identity_residual"]))), "identityBound": got["identityBound"], "maxAbsReferenceMatch": reference_match, "referenceDenominatorMin": float(np.min(d)), "armDenominatorMin": float(np.min(got["Dhat"])), "sumAbsA": got["sumAbsA"], "sumAbsB": got["sumAbsB"], "sumAbsAplusB": got["sumAbsAplusB"], "cancellationC1": got["cancellationC1"], "zeroContributionSum": got["zeroContributionSum"], "oldRelativeL2": old_metric}


def analyze(output: Path, command: list[str], started: datetime) -> dict:
    fixtures = fixture_results()
    dump(output / "fixture_results.json", fixtures)
    all_terms = {name: np.empty((4, 2, 3, 783, 32), dtype=np.float64) for name in ("a", "b", "c", "e", "a_plus_b", "identity_residual", "Nhat")}
    all_denominators = np.empty((4, 2, 3, 783), dtype=np.float64)
    reference_n = np.empty((4, 2, 783, 32), dtype=np.float64)
    reference_d = np.empty((4, 2, 783), dtype=np.float64)
    reference_y = np.empty((4, 2, 783, 32), dtype=np.float64)
    rows, inputs = [], []
    for image in range(4):
        digits = f"{image:05d}"
        trace_dir = ROOT / f"inputs/image_{digits}/trace"
        control_dir = ROOT / f"inputs/image_{digits}/control"
        trace_manifest, control_manifest = verify_manifest(trace_dir), verify_manifest(control_dir)
        metadata, mapped = load_trace(trace_dir)
        control_meta = json.loads((control_dir / "manifest.json").read_text())
        require(control_meta["input_trace"]["manifest_sha256"] == trace_manifest["manifestSha256"], "control input trace manifest mismatch")
        require((control_dir / "trace.json").read_bytes() == (trace_dir / "trace.json").read_bytes(), "control trace copy mismatch")
        old = json.loads((control_dir / "metrics.json").read_text())
        old_rows = {(record["layer"], record["head"], record["arm"]): record for record in old["metrics"]}
        require(set(old_rows) == {(layer, 0, arm) for layer in LAYERS for arm in ARMS}, "old metric arm/selection mismatch")
        with np.load(control_dir / "controls.npz", allow_pickle=False) as archive:
            require(archive["fp64_reference"].shape == SHAPE_VECTOR, "FP64 reference shape mismatch")
            for li, layer in enumerate(LAYERS):
                epsilon = float(metadata["operator"]["epsilonByLayer"][li])
                n, d, y = reference(mapped["q"][:, 0, li, 0], mapped["k"][:, 0, li, 0], mapped["v"][:, 0, li, 0], epsilon)
                saved_reference = archive["fp64_reference"][:, 0, li, 0]
                reference_match = float(np.max(np.abs(y - saved_reference)))
                tolerance = 2e-13 + 2e-13 * float(np.max(np.abs(saved_reference)))
                require(reference_match <= tolerance, "FP64 reference reproduction mismatch")
                reference_n[image, li], reference_d[image, li], reference_y[image, li] = n, d, y
                for ai, arm in enumerate(ARMS):
                    saved_n = archive[f"{arm}_numerator"][:, 0, li, 0]
                    saved_d = archive[f"{arm}_effective_denominator"][:, 0, li, 0]
                    saved_y = archive[f"{arm}_output"][:, 0, li, 0]
                    require(saved_n.shape == (783, 32) and saved_d.shape == (783,) and saved_y.shape == (783, 32), "control array shape mismatch")
                    scale = 0.5 if arm == "bf16_scale_half" else 1.0
                    got = decompose(n, d, y, saved_n, saved_d, saved_y, scale)
                    for name in all_terms:
                        all_terms[name][image, li, ai] = got[name]
                    all_denominators[image, li, ai] = got["Dhat"]
                    rows.append(row_summary(image, layer, arm, reference_match, d, y, got, float(old_rows[(layer, 0, arm)]["vs_fp64"]["relative_l2"])))
        require(verify_manifest(trace_dir) == trace_manifest and verify_manifest(control_dir) == control_manifest, "input manifests changed during analysis")
        inputs.append({"image": image, "trace": trace_manifest, "control": control_manifest, "epsilonBySelectedLayer": metadata["operator"]["epsilonByLayer"]})
    require(len(rows) == 24 and len({(r["image"], r["layer"], r["arm"]) for r in rows}) == 24, "missing or duplicate diagnostic row")
    compare_expected_rows(rows)
    half_comparison = {}
    for name in ("a", "b", "c", "e", "a_plus_b"):
        left, right = all_terms[name][:, :, 1], all_terms[name][:, :, 2]
        half_comparison[name] = {"exactEquality": bool(np.array_equal(left, right)), "maxAbsDifference": float(np.max(np.abs(left - right)))}
    bf16_rows = [row for row in rows if row["arm"] == "bf16"]
    half_c = [row["cancellationC1"] for row in rows if row["arm"] == "bf16_scale_half"]
    bf16_c = [row["cancellationC1"] for row in bf16_rows]
    cancellation_differences = [abs(a - b) if a is not None and b is not None else None for a, b in zip(bf16_c, half_c)]
    half_comparison["cancellationC1"] = {"exactEquality": all(a == b for a, b in zip(bf16_c, half_c)), "maxAbsDifference": max((x for x in cancellation_differences if x is not None), default=None)}
    np.savez(output / "terms.npz", **all_terms, Dhat=all_denominators, N_reference=reference_n, D_reference=reference_d, y_reference=reference_y, arms=np.asarray(ARMS), layers=np.asarray(LAYERS))
    fieldnames = ["image", "layer", "head", "arm", "positions", "coordinatesPerPosition", "scalarCount", "scaleUndone", "referenceNormL2", "referenceNormZero", "relativeNormUndefinedReason", "maxAbsIdentityResidual", "identityBound", "maxAbsReferenceMatch", "referenceDenominatorMin", "armDenominatorMin", "sumAbsA", "sumAbsB", "sumAbsAplusB", "cancellationC1", "zeroContributionSum", "oldRelativeL2"]
    for name in ("a", "b", "a_plus_b", "c", "e"):
        fieldnames += [f"rawNormL2_{name}", f"relativeNormL2_{name}"]
    with (output / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            for name in ("a", "b", "a_plus_b", "c", "e"):
                flat[f"rawNormL2_{name}"] = row["rawNormL2"][name]
                flat[f"relativeNormL2_{name}"] = row["relativeNormL2"][name]
            writer.writerow(flat)
    defined_bf16_c = [value for value in bf16_c if value is not None]
    summary = {"interpretation": "Retrospective descriptive fixed-feature readout decomposition on four observed development images; no full-model score or new-method claim", "rows": rows, "bf16MainRows": bf16_rows, "bf16CancellationRangeAcrossEightSelectedTraces": [min(defined_bf16_c), max(defined_bf16_c)] if defined_bf16_c else None, "bf16UndefinedCancellationRows": len(bf16_c) - len(defined_bf16_c), "halfScaleComparison": half_comparison, "counts": {"images": 4, "selectedLayers": 2, "headsPerLayer": 1, "arms": 3, "rows": 24, "positionsPerImage": 783, "coordinatesPerPosition": 32}, "termsFile": "terms.npz", "rowFile": "rows.csv", "fixtureFile": "fixture_results.json"}
    dump(output / "summary.json", summary)
    provenance = {"source": {"file": "replay.py", "sha256": sha(Path(__file__))}, "contract": {"file": CONTRACT.name, "sha256": sha(CONTRACT)}, "inputs": inputs, "environment": {"pythonVersion": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(), "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"), "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"), "isolatedMode": bool(sys.flags.isolated)}, "commandTemplate": command, "commandIsTemplate": True, "startUtc": started.isoformat(), "endUtc": datetime.now(timezone.utc).isoformat(), "status": "success"}
    dump(output / "provenance.json", provenance)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(f"refusing existing output directory: {output}", file=sys.stderr)
        return 2
    if output.is_relative_to(ROOT):
        print("output must be outside the input bundle", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    tick = time.monotonic()
    command = ["python", "-I", "replay.py", "--output", "NEW_OUTPUT"]
    output.mkdir(parents=True)
    try:
        require(os.getenv("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            require(os.getenv(key) == "2", f"{key} must equal 2")
        require(np.__version__ == "1.26.4", "replay requires pinned NumPy 1.26.4")
        require(sha(CONTRACT) == "97b551835b252296939d3a037d35286a862361a3ea9f2ecef7c30ed19637036b", "frozen contract hash mismatch")
        package_manifest_before = verify_package_manifest()
        summary = analyze(output, command, started)
        require(verify_package_manifest() == package_manifest_before, "package inputs changed during replay")
        (output / "run.log").write_text(f"start_utc={started.isoformat()}\nend_utc={datetime.now(timezone.utc).isoformat()}\nelapsed_seconds={time.monotonic()-tick:.9f}\nexit_code=0\ncommand_template={json.dumps(command)}\ncommand_is_template=true\nisolated_mode={str(bool(sys.flags.isolated)).lower()}\n")
        files = {item.name: {"bytes": item.stat().st_size, "sha256": sha(item)} for item in output.iterdir() if item.is_file() and item.name not in ("manifest.json", "complete.json")}
        dump(output / "manifest.json", {"files": files})
        dump(output / "complete.json", {"status": "complete", "manifestSha256": sha(output / "manifest.json"), "completedUtc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"status": "complete", "rows": len(summary["rows"]), "bf16CancellationRange": summary["bf16CancellationRangeAcrossEightSelectedTraces"], "output": str(output)}))
        return 0
    except Exception as error:
        dump(output / "failure.json", {"status": "failed", "commandTemplate": command, "commandIsTemplate": True, "isolatedMode": bool(sys.flags.isolated), "startUtc": started.isoformat(), "endUtc": datetime.now(timezone.utc).isoformat(), "elapsedSeconds": time.monotonic() - tick, "error": traceback.format_exc(), "sourceSha256": sha(Path(__file__)), "contractSha256": sha(CONTRACT)})
        (output / "run.log").write_text(f"start_utc={started.isoformat()}\nend_utc={datetime.now(timezone.utc).isoformat()}\nelapsed_seconds={time.monotonic()-tick:.9f}\nexit_code=1\nerror={type(error).__name__}: {error}\n")
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
