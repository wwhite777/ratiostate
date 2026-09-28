#!/usr/bin/env python3
"""Score saved browser prediction parameters as sampler-implied discretized mixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METRIC_NAME = "idealized_sampler_implied_conditional_discretized_nll"


class ScoreError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True)), axis=axis)


def component_log_mass(target: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Log probability mass for rounding ((z+1)/2)*255 then clamping to [0,255]."""
    target = np.asarray(target, dtype=np.int64)
    if target.ndim != 1 or ((target < 0) | (target > 255)).any():
        raise ScoreError("targets must be a one-dimensional integer pixel sequence in [0,255]")
    center = 2.0 * target[:, None] / 255.0 - 1.0
    half_width = 1.0 / 255.0
    lo, hi = center - half_width, center + half_width
    a, b = (lo - means) / scales, (hi - means) / scales
    result = log_sigmoid(b) + log_sigmoid(-a) + np.log(-np.expm1(-2.0 / (255.0 * scales)))
    low = target == 0
    high = target == 255
    if low.any(): result[low] = log_sigmoid(b[low])
    if high.any(): result[high] = log_sigmoid(-a[high])
    return result


def score(parameters: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 30 or not np.isfinite(parameters).all():
        raise ScoreError("prediction parameters must be finite shape (steps, 30)")
    targets_array = np.asarray(targets)
    if targets_array.shape != (parameters.shape[0],) or not np.issubdtype(targets_array.dtype, np.integer):
        raise ScoreError("targets must be integer shape (steps,)")
    logits, means, raw_scales = np.split(parameters, 3, axis=1)
    scales = raw_scales.copy()
    negative = raw_scales < 0.0
    scales[negative] = np.exp(raw_scales[negative]) + 0.0001
    scales[~negative] += 1.0001
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ScoreError("invalid mixture scales")
    log_components = component_log_mass(targets_array, means, scales)
    log_weights = logits - logsumexp(logits)[:, None]
    token_log_probability = logsumexp(log_weights + log_components)
    if not np.isfinite(token_log_probability).all():
        raise ScoreError("nonfinite token log probability")
    return {"token_log_probability": token_log_probability, "token_probability": np.exp(token_log_probability), "component_log_mass": log_components, "scales": scales}


def load_run(directory: Path) -> tuple[np.ndarray, dict]:
    for name in ("trace.json", "manifest.json", "complete.json"):
        if not (directory / name).is_file(): raise ScoreError(f"missing {name}")
    manifest, complete = json.loads((directory / "manifest.json").read_text()), json.loads((directory / "complete.json").read_text())
    if complete.get("status") != "complete" or complete.get("manifestSha256", complete.get("manifest_sha256")) != sha(directory / "manifest.json"):
        raise ScoreError("run completion marker is invalid")
    files = manifest.get("files", manifest.get("output_files"))
    if not isinstance(files, dict) or "trace.json" not in files: raise ScoreError("manifest lacks required trace identity")
    root = directory.resolve()
    for name, record in files.items():
        path = (directory / name).resolve()
        if not str(path).startswith(str(root) + "/") or not path.is_file() or not isinstance(record, dict) or record.get("sha256") != sha(path): raise ScoreError(f"manifest mismatch for {name}")
    trace = json.loads((directory / "trace.json").read_text())
    spec = trace.get("arrays", {}).get("prediction_parameters")
    if not isinstance(spec, dict) or spec.get("dtype") != "float32" or spec.get("byteOrder") != "little-endian": raise ScoreError("missing valid prediction_parameters specification")
    shape = tuple(spec.get("shape", [])); file_name = spec.get("file")
    if len(shape) != 3 or shape[1:] != (1, 30) or shape[0] <= 0 or not isinstance(file_name, str) or file_name not in files: raise ScoreError("prediction_parameters must have shape (steps,1,30) and manifest coverage")
    path = (directory / file_name).resolve()
    if not str(path).startswith(str(root) + "/") or path.stat().st_size != 4 * int(np.prod(shape)): raise ScoreError("invalid prediction_parameters file length")
    values = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all(): raise ScoreError("nonfinite prediction parameters")
    input_meta = trace.get("input")
    pixels_sha256 = input_meta.get("pixelsSha256") if isinstance(input_meta, dict) else None
    if not isinstance(input_meta, dict) or input_meta.get("startIndex") != 0 or input_meta.get("steps") != shape[0] or not isinstance(pixels_sha256, str) or len(pixels_sha256) != 64 or any(char not in "0123456789abcdef" for char in pixels_sha256):
        raise ScoreError("run must record startIndex=0 and matching prediction step count")
    return values, {"path":str(root), "trace_sha256":sha(directory/"trace.json"), "manifest_sha256":sha(directory/"manifest.json"), "complete_sha256":sha(directory/"complete.json"), "input":input_meta}


def load_targets(path: Path, steps: int, expected_sha256: str | None) -> np.ndarray:
    document = json.loads(path.read_text())
    pixels = document.get("pixels") if isinstance(document, dict) else document
    if not isinstance(pixels, list) or len(pixels) < steps + 1 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 255 for x in pixels): raise ScoreError("pixels must contain integer targets pixels[1:steps+1]")
    if expected_sha256 is not None and expected_sha256 != sha(path): raise ScoreError("pixels JSON does not match the run input identity")
    return np.asarray(pixels[1:steps + 1], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--pixels-json", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): print(f"refusing existing output directory: {output}", file=sys.stderr); return 2
    started, began = datetime.now(timezone.utc), time.monotonic(); output.mkdir(parents=True)
    try:
        parameters, provenance = load_run(Path(args.run)); targets = load_targets(Path(args.pixels_json), parameters.shape[0], provenance["input"].get("pixelsSha256")); scored = score(parameters[:,0], targets)
        np.savez(output / "token_scores.npz", targets=targets, **scored)
        total_nll = float(-np.sum(scored["token_log_probability"])); bits = total_nll / (math.log(2.0) * len(targets))
        write_json(output / "metrics.json", {"metric":METRIC_NAME, "interpretation":"conditional on saved prediction parameters and pixel 0; sampler-implied discretized mixture metric, not a confirmed training loss or paper BPD reproduction", "scored_pixels":int(len(targets)), "conditioned_pixels":1, "total_nll":total_nll, "bits_per_predicted_pixel":bits, "input_run":provenance, "pixels_json":{"path":str(Path(args.pixels_json).resolve()),"sha256":sha(Path(args.pixels_json))}})
        shutil.copyfile(Path(__file__), output / "score_browser_precision.py"); elapsed=time.monotonic()-began
        (output/"run.log").write_text(f"start_utc={started.isoformat()}\nend_utc={datetime.now(timezone.utc).isoformat()}\nelapsed_seconds={elapsed:.9f}\nexit_code=0\nstatus=success\ncommand={json.dumps([sys.executable,*sys.argv])}\n",encoding="utf-8")
        files={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name not in {"manifest.json","complete.json"}}; write_json(output/"manifest.json",{"files":files,"environment":{"python":sys.executable,"numpy":np.__version__,"platform":platform.platform()}}); write_json(output/"complete.json",{"status":"complete","manifestSha256":sha(output/"manifest.json")}); return 0
    except Exception as exc:
        (output/"run.log").write_text(f"start_utc={started.isoformat()}\nend_utc={datetime.now(timezone.utc).isoformat()}\nelapsed_seconds={time.monotonic()-began:.9f}\nexit_code=1\nstatus=failed\nerror={type(exc).__name__}: {exc}\n",encoding="utf-8"); write_json(output/"failure.json",{"status":"failed","error_type":type(exc).__name__,"error":str(exc)}); print(f"likelihood scoring failed: {exc}",file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
