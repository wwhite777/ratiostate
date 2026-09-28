#!/usr/bin/env python3
"""Private relocated replay of six frozen RatioState inference arms for one image."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("A00", "A10", "A01", "A11", "fp16_boundary", "bf16_pair_boundary")
MODEL_TIMEOUT = 120
SCORE_TIMEOUT = 10
CHILD_BUDGET = 900


def interrupt_handler(signum, _frame):
    raise InterruptedError(f"wrapper received signal {signum}")


class ReplayError(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise ReplayError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def proc_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])


def completed_manifest(directory: Path) -> str:
    manifest_path, complete_path = directory / "manifest.json", directory / "complete.json"
    need(manifest_path.is_file() and complete_path.is_file(), f"missing completion files: {directory}")
    manifest, complete = read(manifest_path), read(complete_path)
    need(complete.get("status") == "complete" and complete.get("manifestSha256") == sha(manifest_path), f"invalid completion marker: {directory}")
    files = manifest.get("files")
    need(isinstance(files, dict) and files, f"empty output manifest: {directory}")
    root = directory.resolve()
    for name, record in files.items():
        file = (directory / name).resolve()
        need(file.is_relative_to(root) and file.is_file() and file.stat().st_size == record.get("bytes") and sha(file) == record.get("sha256"), f"manifest file mismatch: {name}")
    return sha(manifest_path)


def preflight(args) -> tuple[dict, dict, Path, Path, Path, Path]:
    manifest = read(ROOT / "bundle_manifest.json")
    need(manifest.get("status") == "private_source_pack_no_assets" and manifest.get("schemaVersion") == 1, "invalid source-pack manifest")
    need(manifest.get("wrapperSha256") == sha(Path(__file__)), "wrapper source identity mismatch")
    for relative, record in manifest["sourceFiles"].items():
        path = (ROOT / relative).resolve()
        need(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"source identity mismatch: {relative}")
    for relative, record in manifest["runtimeLockFiles"].items():
        path = (ROOT / relative).resolve()
        need(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"packaged runtime lock mismatch: {relative}")
    need(sys.version.split()[0] == "3.10.12", f"Python 3.10.12 required; observed {sys.version.split()[0]}")
    numpy_version = importlib.metadata.version("numpy")
    need(numpy_version == "1.26.4", f"NumPy 1.26.4 required; observed {numpy_version}")
    node = shutil.which(args.node) if args.node else shutil.which("node")
    need(node is not None, "Node executable unavailable")
    version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
    need(version.returncode == 0 and version.stdout.strip() == "v20.19.6", f"Node 20.19.6 required: {version.stdout.strip()} / {version.returncode}")

    model = Path(args.model_dir).resolve()
    need(model.is_dir(), "model-dir is absent")
    for record in manifest["modelFiles"]:
        file = model / record["name"]
        need(file.is_file() and file.stat().st_size == record["bytes"] and sha(file) == record["sha256"], f"model file identity mismatch: {record['name']}")
    pixels = Path(args.pixels_json).resolve()
    match = re.fullmatch(r"test_(\d{5})\.json", pixels.name)
    need(pixels.is_file() and match is not None, "pixels-json must be a test_00000.json file")
    index = int(match.group(1))
    rows = [row for row in manifest["selectedRows"] if row["index"] == index]
    need(len(rows) == 1, "pixel index is outside frozen fixed32 selection")
    row = rows[0]
    need(sha(pixels) == row["pixelsSha256"], "selected pixel JSON hash mismatch")
    document = read(pixels)
    values = document.get("pixels")
    need(document.get("index") == index and document.get("label") == row["label"] and document.get("image_shape") == [28, 28], "selected pixel metadata mismatch")
    need(isinstance(values, list) and len(values) == 784 and all(type(value) is int and 0 <= value <= 255 for value in values), "invalid 784-pixel byte sequence")
    need(document.get("image_source_sha256") == manifest["imageIDXsha256"] and document.get("label_source_sha256") == manifest["labelIDXsha256"], "pixel source identity mismatch")
    need(row["armOrder"][0] == "A00" and set(row["armOrder"]) == set(ARMS) and len(row["armOrder"]) == 6, "invalid frozen six-arm order")

    runtime = Path(args.runtime_dir).resolve()
    need(runtime.is_dir() and (runtime / "node_modules").is_dir(), "runtime-dir is absent or lacks node_modules")
    for relative, record in manifest["runtimeLockFiles"].items():
        name = Path(relative).name
        file = runtime / name
        need(file.is_file() and file.stat().st_size == record["bytes"] and sha(file) == record["sha256"], f"external runtime lock mismatch: {name}")
    tfjs = runtime / "node_modules/@tensorflow/tfjs/package.json"
    need(tfjs.is_file() and read(tfjs).get("version") == "2.0.0", "external TFJS package is not 2.0.0")

    output = Path(args.output_root).resolve()
    need(not output.exists(), "refusing existing output-root")
    return manifest, row, model, pixels, runtime, Path(node).resolve()


def stage_sources(output: Path, manifest: dict, runtime: Path) -> Path:
    stage = output / "runner-root"
    for relative, record in {**manifest["sourceFiles"], **manifest["runtimeLockFiles"]}.items():
        source = ROOT / relative
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        need(sha(destination) == record["sha256"], f"staged source mismatch: {relative}")
    node_modules = stage / ".cache/tfjs_runtime/node_modules"
    node_modules.symlink_to(runtime / "node_modules", target_is_directory=True)
    need(node_modules.is_dir(), "staged external runtime link failed")
    return stage


def command_for(stage: Path, row: dict, arm: str, model: Path, pixels: Path, output: Path, node: Path, native: Path) -> list[str]:
    if arm in ("A00", "A11", "bf16_pair_boundary"):
        script = stage / "scripts/run_browser_precision.cjs"
        cli_arm = {"A00": "native", "A11": "bf16_boundary", "bf16_pair_boundary": "bf16_pair_boundary"}[arm]
    elif arm in ("A10", "A01"):
        script, cli_arm = stage / "scripts/run_state_role_boundary_v1.cjs", arm
    elif arm == "fp16_boundary":
        script, cli_arm = stage / "research/fixed32_20260924/worker/run_fp16_boundary_v3.cjs", arm
        need(completed_manifest(native), "same-image A00 reference incomplete for FP16")
    else:
        raise ReplayError(f"unsupported arm {arm}")
    command = [str(node), "--v8-pool-size=1", str(script), "--model-dir", str(model), "--pixels-json", str(pixels)]
    if arm == "fp16_boundary":
        command += ["--native-reference", str(native)]
    return command + ["--output", str(output), "--steps", "783", "--arm", cli_arm]


def raw_differences(output: Path, expected: dict) -> list[dict]:
    actual = {path.name for path in output.iterdir() if path.is_file() and path.suffix in (".f32", ".i32", ".u16")}
    names = set(expected)
    differences = []
    for name in sorted(names | actual):
        if name not in names:
            differences.append({"file": name, "issue": "unexpected raw file"})
        elif name not in actual:
            differences.append({"file": name, "issue": "missing raw file"})
        else:
            path = output / name
            observed = sha(path)
            if path.stat().st_size != expected[name]["bytes"] or observed != expected[name]["sha256"]:
                differences.append({"file": name, "issue": "raw byte mismatch", "expectedSha256": expected[name]["sha256"], "actualSha256": observed, "expectedBytes": expected[name]["bytes"], "actualBytes": path.stat().st_size})
    return differences


def run_child(ledger: dict, ledger_path: Path, *, kind: str, arm: str, command: list[str], output: Path, timeout: int, env: dict, log_dir: Path) -> dict:
    need(not output.exists(), "refusing existing child output")
    need(ledger["chargedChildWallSeconds"] + timeout <= CHILD_BUDGET, "cumulative child-wall reservation exhausted")
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_dir / f"{kind}_{arm}.stdout.log", log_dir / f"{kind}_{arm}.stderr.log"
    need(not stdout_path.exists() and not stderr_path.exists(), "refusing existing child logs")
    entry = {"kind": kind, "arm": arm, "command": command, "output": str(output), "startedUtc": now(), "status": "launching", "timeoutSeconds": timeout, "reservedSeconds": timeout, "stdoutLog": str(stdout_path), "stderrLog": str(stderr_path)}
    ledger["runs"].append(entry)
    ledger["reservedChildWallSeconds"] = timeout
    write(ledger_path, ledger)
    began = time.monotonic()
    process = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=output.parent.parent, env=env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
            entry.update(status="running", pid=process.pid, processGroupId=process.pid)
            entry["pidStartTicks"] = proc_start_ticks(process.pid)
            write(ledger_path, ledger)
            try:
                code = process.wait(timeout=timeout)
                entry.update(exitCode=code, processReturnCode=code, status="exited" if code == 0 else "nonzero_exit")
            except subprocess.TimeoutExpired:
                entry.update(exitCode=None, status="timeout")
                if process.poll() is None:
                    need(proc_start_ticks(process.pid) == entry["pidStartTicks"] and os.getpgid(process.pid) == process.pid, "owned child identity changed before timeout kill")
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                entry["processReturnCode"] = process.returncode
        entry["elapsedSeconds"] = time.monotonic() - began
        if entry["status"] == "exited":
            entry["manifestSha256"] = completed_manifest(output)
            entry["status"] = "completed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if process is not None:
            entry["processReturnCode"] = process.returncode
        entry.update(status="child_or_validation_error", error=f"{type(error).__name__}: {error}", elapsedSeconds=time.monotonic() - began)
    finally:
        entry.update(endedUtc=now(), elapsedSeconds=time.monotonic() - began,
                     stdoutSha256=sha(stdout_path) if stdout_path.is_file() else None,
                     stderrSha256=sha(stderr_path) if stderr_path.is_file() else None)
        ledger["chargedChildWallSeconds"] += entry["elapsedSeconds"]
        ledger["reservedChildWallSeconds"] = 0
        write(ledger_path, ledger)
    return entry


def execute(args, manifest: dict, row: dict, model: Path, pixels: Path, runtime: Path, node: Path) -> int:
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True)
    ledger_path = output / "ledger.json"
    ledger = {"status": "preparing", "startedUtc": now(), "bundleManifestSha256": sha(ROOT / "bundle_manifest.json"), "sourcePackagePath": str(ROOT), "externalAssets": {"modelDir": str(model), "pixelsJson": str(pixels), "runtimeDir": str(runtime), "runtimeMode": "shared existing supplied runtime; no clean installation"}, "selectedIndex": row["index"], "armOrder": row["armOrder"], "budgetChildWallSeconds": CHILD_BUDGET, "chargedChildWallSeconds": 0.0, "reservedChildWallSeconds": 0.0, "runs": [], "rawComparisons": [], "scoreComparisons": []}
    write(ledger_path, ledger)
    try:
        stage = stage_sources(output, manifest, runtime)
        ledger["stagedSourceRoot"] = str(stage)
        logs = output / "logs"
        logs.mkdir()
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", UV_THREADPOOL_SIZE="1")
        ledger["environmentOverrides"] = {name: env[name] for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "UV_THREADPOOL_SIZE")}
        ledger["status"] = "running"
        write(ledger_path, ledger)
        native = output / "results/A00"
        for arm in row["armOrder"]:
            model_output = output / "results" / arm
            model_command = command_for(stage, row, arm, model, pixels, model_output, node, native)
            model_entry = run_child(ledger, ledger_path, kind="model", arm=arm, command=model_command, output=model_output, timeout=MODEL_TIMEOUT, env=env, log_dir=logs)
            need(model_entry["status"] == "completed", f"model child failed: {arm}/{model_entry['status']}")
            differences = raw_differences(model_output, row["references"][arm]["rawFiles"])
            ledger["rawComparisons"].append({"arm": arm, "checkedFiles": len(row["references"][arm]["rawFiles"]), "differences": differences})
            write(ledger_path, ledger)
            need(not differences, f"raw reproduction differs: {arm}: {len(differences)} files")
            score_output = output / "scores" / arm
            score_command = [sys.executable, str(stage / "scripts/score_browser_precision.py"), "--run", str(model_output), "--pixels-json", str(pixels), "--output", str(score_output)]
            score_entry = run_child(ledger, ledger_path, kind="score", arm=arm, command=score_command, output=score_output, timeout=SCORE_TIMEOUT, env=env, log_dir=logs)
            need(score_entry["status"] == "completed", f"score child failed: {arm}/{score_entry['status']}")
            score = read(score_output / "metrics.json")["bits_per_predicted_pixel"]
            expected = float(row["references"][arm]["bitsPerPredictedPixel"])
            same = type(score) is float and math.isfinite(score) and score == expected
            ledger["scoreComparisons"].append({"arm": arm, "savedBitsPerPredictedPixel": expected, "replayedBitsPerPredictedPixel": score, "exactFloat64Match": same})
            write(ledger_path, ledger)
            need(same, f"saved score mismatch: {arm}")
        ledger["status"] = "complete"
        ledger["endedUtc"] = now()
        write(ledger_path, ledger)
        write(output / "complete.json", {"status": "complete", "ledgerSha256": sha(ledger_path), "completedUtc": now(), "sixModelCalls": True, "sixScoreCalls": True})
        print(json.dumps({"status": "complete", "index": row["index"], "modelCalls": 6, "scoreCalls": 6, "childWallSeconds": ledger["chargedChildWallSeconds"], "ledgerSha256": sha(ledger_path)}))
        return 0
    except BaseException as error:
        ledger["status"] = "failed_or_partial"
        ledger["endedUtc"] = now()
        ledger["error"] = f"{type(error).__name__}: {error}"
        write(ledger_path, ledger)
        write(output / "failure.json", {"status": "failed_or_partial", "error": ledger["error"], "ledgerSha256": sha(ledger_path)})
        print(ledger["error"], file=sys.stderr)
        return 1


def main() -> int:
    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pixels-json", required=True)
    parser.add_argument("--runtime-dir", required=True, help="existing folder containing package-lock.json and node_modules")
    parser.add_argument("--output-root", required=True, help="fresh output directory; existing paths are refused")
    parser.add_argument("--node", help="Node 20.19.6 executable; defaults to node on PATH")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="validate sources/assets without inference or output creation")
    mode.add_argument("--execute", action="store_true", help="run six saved-image reproduction cells")
    args = parser.parse_args()
    try:
        manifest, row, model, pixels, runtime, node = preflight(args)
        if args.check_only:
            print(json.dumps({"status": "preflight_pass_no_inference", "selectedIndex": row["index"], "sourceManifestSha256": sha(ROOT / "bundle_manifest.json"), "modelGraphSha256": manifest["graphSha256"], "runtimeMode": "external supplied existing runtime"}))
            return 0
        return execute(args, manifest, row, model, pixels, runtime, node)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
