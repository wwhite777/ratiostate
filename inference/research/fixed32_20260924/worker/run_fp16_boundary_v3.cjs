#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const os = require("os");

const ROOT = path.resolve(__dirname, "../../..");
const TFJS_ROOT = path.join(ROOT, ".cache", "tfjs_runtime", "node_modules", "@tensorflow", "tfjs");
const EXPECTED_GRAPH_SHA256 = "304f1905c945d25b4848873feaa3ea04c1fa6c84fa9230b214c24d2b85b1f83e";
const EXPECTED_WEIGHT_BYTES = 25839100;
const LAYERS = 8, HEADS = 8, WIDTH = 32;
const KC_ELEMENTS = LAYERS * HEADS * WIDTH;
const KV_ELEMENTS = LAYERS * HEADS * WIDTH * WIDTH;
const CACHE_ELEMENTS = KC_ELEMENTS + KV_ELEMENTS;
const SELECTED_LAYERS = [0, 7], SELECTED_HEAD = 0;
const ARMS = new Set(["fp16_boundary", "native_check"]);

function sha256File(filename) {
  const hash = crypto.createHash("sha256"); hash.update(fs.readFileSync(filename)); return hash.digest("hex");
}

function writeJson(filename, value) { fs.writeFileSync(filename, JSON.stringify(value, null, 2) + "\n", "utf8"); }

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index].startsWith("--") || index + 1 >= argv.length) throw new Error("arguments must be --name value pairs");
    args[argv[index].slice(2)] = argv[index + 1];
  }
  for (const name of ["model-dir", "pixels-json", "native-reference", "output", "steps", "arm"]) if (!(name in args)) throw new Error(`missing --${name}`);
  if (!ARMS.has(args.arm)) throw new Error(`arm must be one of ${Array.from(ARMS).join(", ")}`);
  args.steps = Number(args.steps);
  if (!Number.isInteger(args.steps) || args.steps <= 0 || args.steps > 783) throw new Error("steps must be an integer in [1, 783]");
  return args;
}

function loadPixels(filename, steps) {
  const document = JSON.parse(fs.readFileSync(filename, "utf8"));
  const pixels = Array.isArray(document) ? document : document && document.pixels;
  if (!Array.isArray(pixels) || pixels.length < steps || pixels.some((value) => !Number.isInteger(value) || value < 0 || value > 255)) {
    throw new Error("pixels JSON must contain one integer sequence in [0,255] with at least steps values");
  }
  if (Array.isArray(pixels[0])) throw new Error("precision runner requires one image per invocation");
  return pixels;
}

function loadModelArtifacts(modelDir) {
  const graphPath = path.join(modelDir, "model.json");
  if (!fs.existsSync(graphPath)) throw new Error(`missing model graph: ${graphPath}`);
  const graphSha256 = sha256File(graphPath);
  if (graphSha256 !== EXPECTED_GRAPH_SHA256) throw new Error(`model graph hash mismatch: ${graphSha256}`);
  const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
  if (graph.format !== "graph-model" || graph.generatedBy !== "2.3.0-rc0" || graph.convertedBy !== "TensorFlow.js Converter v2.0.1") throw new Error("unexpected graph identity");
  const weightSpecs = [], buffers = [], files = [{path: "model.json", bytes: fs.statSync(graphPath).size, sha256: graphSha256}];
  let weightBytes = 0;
  for (const group of graph.weightsManifest || []) {
    weightSpecs.push(...group.weights);
    for (const relative of group.paths) {
      const filename = path.resolve(modelDir, relative);
      if (!filename.startsWith(path.resolve(modelDir) + path.sep) || !fs.existsSync(filename)) throw new Error(`missing or unsafe weight path: ${relative}`);
      const data = fs.readFileSync(filename); buffers.push(data); weightBytes += data.length;
      files.push({path: relative, bytes: data.length, sha256: sha256File(filename)});
    }
  }
  if (weightBytes !== EXPECTED_WEIGHT_BYTES) throw new Error(`model weight byte total mismatch: ${weightBytes}`);
  const combined = Buffer.concat(buffers);
  return {
    artifacts: {modelTopology: graph.modelTopology, weightSpecs, weightData: combined.buffer.slice(combined.byteOffset, combined.byteOffset + combined.byteLength), format: graph.format, generatedBy: graph.generatedBy, convertedBy: graph.convertedBy, userDefinedMetadata: graph.userDefinedMetadata},
    identity: {graphSha256, weightBytes, files},
  };
}

// The codec consumes FP32 bits directly, so signed zeros and NaN classification are explicit.
function float32BitsToFloat16Word(bits) {
  const sign = (bits >>> 16) & 0x8000, exponent = (bits >>> 23) & 255, fraction = bits & 0x7fffff;
  if (exponent === 255) return sign | (fraction === 0 ? 0x7c00 : 0x7e00 | ((fraction >>> 13) & 0x01ff));
  if (exponent === 0) return sign;
  const e = exponent - 127;
  if (e > 15) return sign | 0x7c00;
  if (e >= -14) {
    let mantissa = fraction >>> 13;
    const remainder = fraction & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (mantissa & 1))) mantissa++;
    let halfExponent = e + 15;
    if (mantissa === 1024) { mantissa = 0; halfExponent++; }
    return sign | (halfExponent << 10) | mantissa;
  }
  if (e < -25) return sign;
  const significand = 0x800000 | fraction, divisor = 2 ** (-e - 1);
  let mantissa = Math.floor(significand / divisor);
  const remainder = significand - mantissa * divisor;
  if (remainder > divisor / 2 || (remainder === divisor / 2 && (mantissa & 1))) mantissa++;
  return sign | mantissa;
}
function float16WordToFloat32(word) {
  const sign = word & 0x8000 ? -1 : 1, exponent = (word >>> 10) & 31, fraction = word & 1023;
  if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}
function encodeFp16(values) {
  if (!(values instanceof Float32Array)) throw new Error("FP16 encoder needs Float32Array");
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length), out = new Uint16Array(values.length);
  for (let index = 0; index < out.length; index++) out[index] = float32BitsToFloat16Word(bits[index]);
  return out;
}
function decodeFp16(words) {
  if (!(words instanceof Uint16Array)) throw new Error("FP16 decoder needs Uint16Array");
  const out = new Float32Array(words.length);
  for (let index = 0; index < out.length; index++) out[index] = float16WordToFloat32(words[index]);
  return out;
}
function writeSelected(target, data, step, width, secondWidth = 1) {
  let destination = step * SELECTED_LAYERS.length * width * secondWidth;
  for (const layer of SELECTED_LAYERS) {
    const base = ((layer * HEADS + SELECTED_HEAD) * width) * secondWidth;
    const count = width * secondWidth;
    target.set(data.subarray(base, base + count), destination);
    destination += count;
  }
}
function writeRaw(filename, values, type) {
  if (os.endianness() !== "LE" || !(values instanceof type)) throw new Error("raw writer requires little-endian typed array");
  fs.writeFileSync(filename, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}
function countRetained(kc, kv) {
  const seen = new Set([kc.buffer, kv.buffer]);
  let bytes = 0; for (const buffer of seen) bytes += buffer.byteLength;
  return {bytes, backingBuffers: seen.size, viewByteLengths: [kc.byteLength, kv.byteLength]};
}
function outputManifest(outputDir) {
  const files = {};
  for (const entry of fs.readdirSync(outputDir).sort()) {
    const filename = path.join(outputDir, entry);
    if (fs.statSync(filename).isFile() && entry !== "manifest.json" && entry !== "complete.json") files[entry] = {bytes: fs.statSync(filename).size, sha256: sha256File(filename)};
  }
  return {files};
}
function failureAt(kind, token, state, index, value, word, range) {
  const error = new Error(`${kind} at token=${token} state=${state} index=${index} value=${value} word=${word === null ? "n/a" : word.toString(16)}`);
  error.arithmetic = {kind, token, state, index, value: Number.isFinite(value) ? value : String(value), word, range};
  return error;
}
function encodeChecked(values, state, token, range) {
  const words = encodeFp16(values), bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  let firstFailure = null;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (!Number.isFinite(x)) {
      range.nonfiniteCount++;
      if (!firstFailure) firstFailure = {kind: "nonfinite_graph_state", token, state, index: i, value: x, word: null};
      continue;
    }
    const magnitude = Math.abs(x);
    if (magnitude > range.maxAbsReturned) range.maxAbsReturned = magnitude;
    if (x < range.minReturned) range.minReturned = x;
    if (x > range.maxReturned) range.maxReturned = x;
    if ((words[i] & 0x7c00) === 0x7c00) {
      range.overflowCount++;
      if (!firstFailure) firstFailure = {kind: "finite_fp16_overflow", token, state, index: i, value: x, word: words[i], float32Bits: bits[i] >>> 0};
    }
  }
  return {words, firstFailure};
}
function traceBuffers(steps) {
  const kcSelected = steps * SELECTED_LAYERS.length * WIDTH, kvSelected = kcSelected * WIDTH;
  return {
    prediction_parameters: new Float32Array(steps * 30),
    native_kc_state: new Float32Array(kcSelected), native_kv_state: new Float32Array(kvSelected),
    feed_kc_state: new Float32Array(kcSelected), feed_kv_state: new Float32Array(kvSelected),
    stored_fp16_kc: new Uint16Array(kcSelected), stored_fp16_kv: new Uint16Array(kvSelected),
  };
}
function saveTrace(outputDir, trace, steps, complete) {
  const selected = SELECTED_LAYERS.length;
  const shapes = {
    prediction_parameters: [steps, 1, 30],
    native_kc_state: [steps, 1, selected, 1, WIDTH], native_kv_state: [steps, 1, selected, 1, WIDTH, WIDTH],
    feed_kc_state: [steps, 1, selected, 1, WIDTH], feed_kv_state: [steps, 1, selected, 1, WIDTH, WIDTH],
    stored_fp16_kc: [steps, 1, selected, 1, WIDTH], stored_fp16_kv: [steps, 1, selected, 1, WIDTH, WIDTH],
  };
  const arrays = {};
  for (const [name, values] of Object.entries(trace)) {
    const isWord = values instanceof Uint16Array;
    const file = `${complete ? "" : "partial_"}${name}.${isWord ? "u16" : "f32"}`;
    const count = shapes[name].reduce((a, b) => a * b, 1);
    if (values.length !== count) throw new Error(`trace length mismatch ${name}`);
    writeRaw(path.join(outputDir, file), values, isWord ? Uint16Array : Float32Array);
    arrays[name] = {file, dtype: isWord ? "uint16" : "float32", byteOrder: "little-endian", shape: shapes[name]};
  }
  return arrays;
}
function saveTokenZero(outputDir, nativeKc, nativeKv, wordsKc, wordsKv, feedKc, feedKv) {
  const entries = {
    token0_native_kc_state: nativeKc, token0_native_kv_state: nativeKv,
    token0_stored_kc: wordsKc, token0_stored_kv: wordsKv,
    token0_feed_kc_state: feedKc, token0_feed_kv_state: feedKv,
  };
  for (const [name, values] of Object.entries(entries)) writeRaw(path.join(outputDir, `${name}.${values instanceof Uint16Array ? "u16" : "f32"}`), values, values instanceof Uint16Array ? Uint16Array : Float32Array);
}
function validatedNativeReference(directory, pixelsSha256) {
  const root = path.resolve(directory), manifestPath = path.join(root, "manifest.json"), completePath = path.join(root, "complete.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")), complete = JSON.parse(fs.readFileSync(completePath, "utf8"));
  if (complete.status !== "complete" || complete.manifestSha256 !== sha256File(manifestPath) || !manifest.files || typeof manifest.files !== "object") throw new Error("invalid native reference completion");
  for (const [name, record] of Object.entries(manifest.files)) {
    const filename = path.resolve(root, name);
    if (!filename.startsWith(root + path.sep) || !fs.statSync(filename).isFile() || fs.statSync(filename).size !== record.bytes || sha256File(filename) !== record.sha256) throw new Error(`native reference manifest mismatch: ${name}`);
  }
  const trace = JSON.parse(fs.readFileSync(path.join(root, "trace.json"), "utf8"));
  const spec = trace.arrays && trace.arrays.prediction_parameters;
  if (trace.arm !== "native" || trace.model.graphSha256 !== EXPECTED_GRAPH_SHA256 || trace.input.pixelsSha256 !== pixelsSha256 || trace.input.steps !== 783 || trace.input.startIndex !== 0 || !spec || JSON.stringify(spec.shape) !== JSON.stringify([783, 1, 30]) || spec.dtype !== "float32" || spec.byteOrder !== "little-endian" || spec.file !== "prediction_parameters.f32" || !manifest.files[spec.file]) throw new Error("native reference identity/schema mismatch");
  const predictionPath = path.join(root, spec.file), bytes = fs.readFileSync(predictionPath);
  if (bytes.length !== 783 * 30 * 4) throw new Error("native reference prediction bytes mismatch");
  return {bytes, directory: root, manifestSha256: sha256File(manifestPath), predictionSha256: sha256File(predictionPath)};
}
async function runNativeCheck(tf, model, pixels, outputDir, image, nativeReference) {
  const expected = nativeReference.bytes;
  let kc = tf.zeros([1, LAYERS, HEADS, WIDTH], "float32"), kv = tf.zeros([1, LAYERS, HEADS, WIDTH, WIDTH], "float32");
  const saved = new Float32Array(4 * 30);
  for (let step = 0; step < 4; step++) {
    const i = tf.tensor1d([step], "int32"), x = tf.tensor2d([pixels[step]], [1, 1], "int32");
    const [prediction, nextKv, nextKc] = model.execute({i, x, kc, kv}, ["Identity", "Identity_1", "Identity_2"]);
    saved.set(prediction.dataSync(), step * 30);
    kc.dispose(); kv.dispose(); prediction.dispose(); i.dispose(); x.dispose();
    kc = nextKc; kv = nextKv;
  }
  kc.dispose(); kv.dispose();
  const actual = Buffer.from(saved.buffer, saved.byteOffset, saved.byteLength);
  if (!actual.equals(expected.subarray(0, actual.length))) throw new Error("short original-graph native prediction reference mismatch");
  writeRaw(path.join(outputDir, "native_check_prediction_parameters.f32"), saved, Float32Array);
  return {image, calls: 4, predictionExact: true, referencePredictionSha256: nativeReference.predictionSha256};
}
async function runFp16(tf, model, pixels, outputDir, steps, image, nativeReference) {
  let kcWords = new Uint16Array(KC_ELEMENTS), kvWords = new Uint16Array(KV_ELEMENTS);
  const retained = countRetained(kcWords, kvWords), trace = traceBuffers(steps);
  const range = {maxAbsReturned: 0, minReturned: Infinity, maxReturned: -Infinity, nonfiniteCount: 0, overflowCount: 0, inspectedCalls: 0};
  let completed = 0;
  try {
    for (let step = 0; step < steps; step++) {
      const kc = tf.tensor(decodeFp16(kcWords), [1, LAYERS, HEADS, WIDTH], "float32");
      const kv = tf.tensor(decodeFp16(kvWords), [1, LAYERS, HEADS, WIDTH, WIDTH], "float32");
      const i = tf.tensor1d([step], "int32"), x = tf.tensor2d([pixels[step]], [1, 1], "int32");
      let prediction, nativeKvTensor, nativeKcTensor;
      try {
        [prediction, nativeKvTensor, nativeKcTensor] = model.execute({i, x, kc, kv}, ["Identity", "Identity_1", "Identity_2"]);
        if (prediction.size !== 30 || nativeKcTensor.size !== KC_ELEMENTS || nativeKvTensor.size !== KV_ELEMENTS || prediction.dtype !== "float32" || nativeKcTensor.dtype !== "float32" || nativeKvTensor.dtype !== "float32") throw new Error("unexpected graph output shape/dtype");
        const p = prediction.dataSync(), nkc = nativeKcTensor.dataSync(), nkv = nativeKvTensor.dataSync();
        for (let j = 0; j < p.length; j++) if (!Number.isFinite(p[j])) throw failureAt("nonfinite_prediction", step, "prediction", j, p[j], null, range);
        if (step === 0) {
          const firstNative = nativeReference.bytes;
          const firstBytes = Buffer.from(p.buffer, p.byteOffset, p.byteLength);
          if (!firstBytes.equals(firstNative.subarray(0, firstBytes.length))) throw new Error("first-call prediction differs from accepted native reference");
          writeRaw(path.join(outputDir, "token0_native_kc_state.f32"), nkc, Float32Array);
          writeRaw(path.join(outputDir, "token0_native_kv_state.f32"), nkv, Float32Array);
        }
        trace.prediction_parameters.set(p, step * 30);
        writeSelected(trace.native_kc_state, nkc, step, WIDTH);
        writeSelected(trace.native_kv_state, nkv, step, WIDTH, WIDTH);
        const encodedKc = encodeChecked(nkc, "kc", step, range), encodedKv = encodeChecked(nkv, "kv", step, range);
        range.inspectedCalls = step + 1;
        const issue = encodedKc.firstFailure || encodedKv.firstFailure;
        if (issue) throw failureAt(issue.kind, issue.token, issue.state, issue.index, issue.value, issue.word, {...range, float32Bits: issue.float32Bits ?? null});
        const nextKc = encodedKc.words, nextKv = encodedKv.words;
        const feedKc = decodeFp16(nextKc), feedKv = decodeFp16(nextKv);
        writeSelected(trace.feed_kc_state, feedKc, step, WIDTH);
        writeSelected(trace.feed_kv_state, feedKv, step, WIDTH, WIDTH);
        writeSelected(trace.stored_fp16_kc, nextKc, step, WIDTH);
        writeSelected(trace.stored_fp16_kv, nextKv, step, WIDTH, WIDTH);
        if (step === 0) saveTokenZero(outputDir, nkc, nkv, nextKc, nextKv, feedKc, feedKv);
        kcWords = nextKc; kvWords = nextKv; completed++; range.inspectedCalls = completed;
      } finally {
        kc.dispose(); kv.dispose(); i.dispose(); x.dispose();
        if (prediction) prediction.dispose();
        if (nativeKcTensor) nativeKcTensor.dispose();
        if (nativeKvTensor) nativeKvTensor.dispose();
      }
    }
    const teacher = Buffer.allocUnsafe(steps * 4);
    for (let j = 0; j < steps; j++) teacher.writeInt32LE(pixels[j], j * 4);
    fs.writeFileSync(path.join(outputDir, "teacher_inputs.i32"), teacher);
    const arrays = saveTrace(outputDir, trace, steps, true);
    return {arrays, range, retained, tfMemoryAtBoundary: tf.memory(), completedCalls: completed};
  } catch (error) {
    const partial = {};
    for (const [name, values] of Object.entries(trace)) {
      const unit = name === "prediction_parameters" ? 30 : name.includes("kc") ? 2 * WIDTH : 2 * WIDTH * WIDTH;
      partial[name] = values.subarray(0, completed * unit);
    }
    const lengths = Object.fromEntries(Object.entries(partial).map(([name, value]) => [name, value.length]));
    for (const [name, values] of Object.entries(partial)) writeRaw(path.join(outputDir, `partial_${name}.${values instanceof Uint16Array ? "u16" : "f32"}`), values, values instanceof Uint16Array ? Uint16Array : Float32Array);
    error.partial = {completedCalls: completed, range, lengths, retained};
    throw error;
  }
}
async function main() {
  let args; try { args = parseArgs(process.argv.slice(2)); } catch (error) { console.error(error.message); return 2; }
  const outputDir = path.resolve(args.output);
  if (fs.existsSync(outputDir)) { console.error(`refusing existing output directory: ${outputDir}`); return 2; }
  const started = new Date(), tick = process.hrtime.bigint(), command = [process.execPath, ...process.execArgv, ...process.argv.slice(1)];
  fs.mkdirSync(outputDir, {recursive: true});
  try {
    if (process.env.CUDA_VISIBLE_DEVICES !== "" || process.env.OMP_NUM_THREADS !== "1" || process.env.OPENBLAS_NUM_THREADS !== "1" || process.env.MKL_NUM_THREADS !== "1" || process.env.UV_THREADPOOL_SIZE !== "1" || !process.execArgv.includes("--v8-pool-size=1")) throw new Error("CPU execution caps not set");
    if (os.endianness() !== "LE") throw new Error("little-endian machine required");
    const modelDir = path.resolve(args["model-dir"]), pixelFile = path.resolve(args["pixels-json"]);
    const image = Number(path.basename(pixelFile).match(/^test_(\d{5})\.json$/)?.[1]);
    if (!Number.isInteger(image) || image < 0 || image > 9999) throw new Error("pixel JSON filename must encode a valid MNIST test index");
    if (args.arm === "fp16_boundary" && args.steps !== 783) throw new Error("FP16 natural run requires 783 steps");
    if (args.arm === "native_check" && args.steps !== 4) throw new Error("native check requires 4 steps");
    const pixels = loadPixels(pixelFile, args.steps), pixelSha256 = sha256File(pixelFile), nativeReference = validatedNativeReference(args["native-reference"], pixelSha256), loaded = loadModelArtifacts(modelDir);
    const tf = require(TFJS_ROOT); await tf.setBackend("cpu"); await tf.ready();
    if (tf.version.tfjs !== "2.0.0" || tf.getBackend() !== "cpu") throw new Error("unexpected TFJS runtime/backend");
    const model = await tf.loadGraphModel(tf.io.fromMemory(loaded.artifacts));
    const tfMemoryBefore = tf.memory();
    const runResult = args.arm === "native_check" ? await runNativeCheck(tf, model, pixels, outputDir, image, nativeReference) : await runFp16(tf, model, pixels, outputDir, args.steps, image, nativeReference);
    const tfMemoryBeforeDispose = tf.memory(); model.dispose();
    if (args.arm === "fp16_boundary") {
      const trace = {contract: "IEEE binary16 Uint16 retained boundary storage; unmodified full graph FP32 update/readout/prediction", arm: args.arm, armDefinition: "returned states rounded RN-even to IEEE binary16 after current readout; Uint16 words retained and decoded to FP32 tensors before next call", model: loaded.identity, input: {pixelsPath: pixelFile, pixelsSha256: pixelSha256, batch: 1, steps: args.steps, startIndex: 0, positions: [0, args.steps - 1], ordering: "teacher input pixels[t] with graph position i=t from zero state"}, nativeReference: {manifestSha256: nativeReference.manifestSha256, predictionSha256: nativeReference.predictionSha256}, runtime: {tfjs: tf.version.tfjs, backend: tf.getBackend(), node: process.version}, selection: {layers: SELECTED_LAYERS, heads: [SELECTED_HEAD]}, storage: {cacheElements: CACHE_ELEMENTS, physicalRetainedPayloadBytesBatch1: runResult.retained.bytes, backingBuffers: runResult.retained.backingBuffers, actualFormat: "Uint16Array IEEE binary16", caveat: "FP32 transient tensors are required; retained bytes do not imply peak-memory or RSS savings"}, arrays: runResult.arrays, teacherInputs: {file: "teacher_inputs.i32", dtype: "int32", byteOrder: "little-endian", shape: [args.steps, 1]}, tokenZeroFullStateFiles: ["token0_native_kc_state.f32", "token0_native_kv_state.f32", "token0_stored_kc.u16", "token0_stored_kv.u16", "token0_feed_kc_state.f32", "token0_feed_kv_state.f32"], range: runResult.range, randomSamplingExecuted: false};
      writeJson(path.join(outputDir, "trace.json"), trace);
    }
    const elapsed = Number(process.hrtime.bigint() - tick) / 1e9;
    writeJson(path.join(outputDir, "run.json"), {startUtc: started.toISOString(), endUtc: new Date().toISOString(), elapsedSeconds: elapsed, exitCode: 0, command, arm: args.arm, source: {runnerSha256: sha256File(__filename), predecessorRunnerSha256: sha256File(path.join(ROOT, "scripts/run_fp16_boundary_v2.cjs")), contractSha256: sha256File(path.join(ROOT, "research/FIXED32_EVALUATION_CONTRACT_v2.md")), scorerSha256: sha256File(path.join(ROOT, "scripts/score_browser_precision.py"))}, model: loaded.identity, input: {pixelFile, sha256: pixelSha256, image, steps: args.steps}, nativeReference: {directory: nativeReference.directory, manifestSha256: nativeReference.manifestSha256, predictionSha256: nativeReference.predictionSha256}, runtime: {node: process.version, tfjs: tf.version.tfjs, backend: tf.getBackend(), tfMemoryBefore, tfMemoryBeforeDispose}, result: runResult});
    fs.copyFileSync(__filename, path.join(outputDir, "run_fp16_boundary_v3.cjs"));
    fs.writeFileSync(path.join(outputDir, "run.log"), `start_utc=${started.toISOString()}\nend_utc=${new Date().toISOString()}\nelapsed_seconds=${elapsed.toFixed(9)}\nexit_code=0\ncommand=${JSON.stringify(command)}\n`);
    writeJson(path.join(outputDir, "manifest.json"), outputManifest(outputDir));
    writeJson(path.join(outputDir, "complete.json"), {status: "complete", manifestSha256: sha256File(path.join(outputDir, "manifest.json")), completedUtc: new Date().toISOString()});
    console.log(`${args.arm} image=${image} complete`); return 0;
  } catch (error) {
    const elapsed = Number(process.hrtime.bigint() - tick) / 1e9;
    const failure = {startUtc: started.toISOString(), endUtc: new Date().toISOString(), elapsedSeconds: elapsed, exitCode: 1, command, error: error.stack || String(error), arithmetic: error.arithmetic || null, partial: error.partial || null, runnerSha256: sha256File(__filename)};
    writeJson(path.join(outputDir, "failure.json"), failure);
    fs.copyFileSync(__filename, path.join(outputDir, "run_fp16_boundary_v3.cjs"));
    fs.writeFileSync(path.join(outputDir, "run.log"), `start_utc=${started.toISOString()}\nend_utc=${new Date().toISOString()}\nelapsed_seconds=${elapsed.toFixed(9)}\nexit_code=1\nerror=${error.message}\n`);
    console.error(error.stack || error.message); return 1;
  }
}
if (require.main === module) main().then(code => { process.exitCode = code; });
module.exports = {float32BitsToFloat16Word, float16WordToFloat32, encodeFp16, decodeFp16, countRetained, KC_ELEMENTS, KV_ELEMENTS, CACHE_ELEMENTS};
