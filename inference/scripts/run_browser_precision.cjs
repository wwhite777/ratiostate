#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const os = require("os");

const ROOT = path.resolve(__dirname, "..");
const TFJS_ROOT = path.join(ROOT, ".cache", "tfjs_runtime", "node_modules", "@tensorflow", "tfjs");
const EXPECTED_GRAPH_SHA256 = "304f1905c945d25b4848873feaa3ea04c1fa6c84fa9230b214c24d2b85b1f83e";
const EXPECTED_WEIGHT_BYTES = 25839100;
const LAYERS = 8, HEADS = 8, WIDTH = 32;
const KC_ELEMENTS = LAYERS * HEADS * WIDTH;
const KV_ELEMENTS = LAYERS * HEADS * WIDTH * WIDTH;
const CACHE_ELEMENTS = KC_ELEMENTS + KV_ELEMENTS;
const SELECTED_LAYERS = [0, 7], SELECTED_HEAD = 0;
const ARMS = new Set(["native", "bf16_boundary", "bf16_pair_boundary"]);

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
  for (const name of ["model-dir", "pixels-json", "output", "steps", "arm"]) if (!(name in args)) throw new Error(`missing --${name}`);
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

function roundFloat32ToBf16InPlace(values) {
  if (!(values instanceof Float32Array)) throw new Error("BF16 encoder requires Float32Array");
  const words = new Uint32Array(values.buffer, values.byteOffset, values.length);
  for (let index = 0; index < words.length; index++) {
    const word = words[index] >>> 0;
    const magnitude = word & 0x7fffffff;
    if (magnitude > 0x7f800000) {
      words[index] = ((word & 0xffff0000) | 0x00010000) >>> 0;
    } else {
      const lsb = (word >>> 16) & 1;
      words[index] = ((word + 0x7fff + lsb) & 0xffff0000) >>> 0;
    }
  }
  return values;
}

function roundFloat32ToBf16(values) { return roundFloat32ToBf16InPlace(new Float32Array(values)); }

function encodeBoundary(nativeValues, previousFeed, arm) {
  if (nativeValues.length !== previousFeed.length) throw new Error("state length changed")
  let high, low = null, feed;
  if (arm === "native") {
    high = nativeValues; feed = nativeValues;
  } else {
    high = roundFloat32ToBf16(nativeValues);
    if (arm === "bf16_boundary") {
      feed = high;
    } else {
      const residual = new Float32Array(nativeValues.length);
      for (let index = 0; index < residual.length; index++) residual[index] = Math.fround(nativeValues[index] - high[index]);
      low = roundFloat32ToBf16(residual);
      feed = new Float32Array(nativeValues.length);
      for (let index = 0; index < feed.length; index++) feed[index] = Math.fround(high[index] + low[index]);
    }
  }
  let changed = 0, lostUpdate = 0, nonzeroUpdate = 0, maxAbs = 0, sumSquares = 0, highMaxAbs = 0;
  for (let index = 0; index < feed.length; index++) {
    const updateChanged = nativeValues[index] !== previousFeed[index];
    if (updateChanged) nonzeroUpdate++;
    if (updateChanged && feed[index] === previousFeed[index]) lostUpdate++;
    if (feed[index] !== nativeValues[index]) changed++;
    const error = feed[index] - nativeValues[index];
    const absolute = Math.abs(error); if (absolute > maxAbs) maxAbs = absolute; sumSquares += error * error;
    const highAbsolute = Math.abs(high[index] - nativeValues[index]); if (highAbsolute > highMaxAbs) highMaxAbs = highAbsolute;
  }
  return {high, low, feed, summary: {elements: feed.length, nonzeroUpdateCount: nonzeroUpdate, lostUpdateCount: lostUpdate, boundaryChangedCount: changed, boundaryErrorMaxAbs: maxAbs, boundaryErrorL2: Math.sqrt(sumSquares), highWordErrorMaxAbs: highMaxAbs}};
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

function writeFloat32(filename, values) {
  if (os.endianness() !== "LE" || !(values instanceof Float32Array)) throw new Error("raw writer requires little-endian Float32Array");
  fs.writeFileSync(filename, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function allFinite(values) { for (let index = 0; index < values.length; index++) if (!Number.isFinite(values[index])) return false; return true; }

function outputManifest(outputDir, excluded) {
  const files = {};
  for (const entry of fs.readdirSync(outputDir).sort()) {
    const filename = path.join(outputDir, entry);
    if (fs.statSync(filename).isFile() && !excluded.has(entry)) files[entry] = {bytes: fs.statSync(filename).size, sha256: sha256File(filename)};
  }
  return files;
}

async function run(args, outputDir) {
  const tf = require(TFJS_ROOT); await tf.setBackend("cpu"); await tf.ready();
  if (tf.version.tfjs !== "2.0.0" || tf.getBackend() !== "cpu") throw new Error(`unexpected TFJS runtime/backend: ${tf.version.tfjs}/${tf.getBackend()}`);
  const modelDir = path.resolve(args["model-dir"]), pixelsPath = path.resolve(args["pixels-json"]);
  const pixels = loadPixels(pixelsPath, args.steps), loaded = loadModelArtifacts(modelDir);
  const model = await tf.loadGraphModel(tf.io.fromMemory(loaded.artifacts));
  let kc = tf.zeros([1, LAYERS, HEADS, WIDTH], "float32"), kv = tf.zeros([1, LAYERS, HEADS, WIDTH, WIDTH], "float32");
  let previousKcFeed = new Float32Array(KC_ELEMENTS), previousKvFeed = new Float32Array(KV_ELEMENTS);
  const kcSelectedCount = args.steps * SELECTED_LAYERS.length * WIDTH;
  const kvSelectedCount = args.steps * SELECTED_LAYERS.length * WIDTH * WIDTH;
  const trace = {
    prediction: new Float32Array(args.steps * 30),
    nativeKc: new Float32Array(kcSelectedCount), nativeKv: new Float32Array(kvSelectedCount),
    feedKc: new Float32Array(kcSelectedCount), feedKv: new Float32Array(kvSelectedCount),
    highKc: new Float32Array(kcSelectedCount), highKv: new Float32Array(kvSelectedCount),
    lowKc: args.arm === "bf16_pair_boundary" ? new Float32Array(kcSelectedCount) : null,
    lowKv: args.arm === "bf16_pair_boundary" ? new Float32Array(kvSelectedCount) : null,
  };
  const summaries = [];
  const tfMemoryBefore = tf.memory();
  const loopBegan = process.hrtime.bigint();
  for (let step = 0; step < args.steps; step++) {
    const iTensor = tf.tensor1d([step], "int32"), xTensor = tf.tensor2d([pixels[step]], [1, 1], "int32");
    const outputs = model.execute({i: iTensor, x: xTensor, kc, kv}, ["Identity", "Identity_1", "Identity_2"]);
    if (!Array.isArray(outputs) || outputs.length !== 3) throw new Error("graph returned an unexpected output set");
    const [prediction, nativeKvTensor, nativeKcTensor] = outputs;
    if (prediction.dtype !== "float32" || prediction.shape.join(",") !== "1,30") throw new Error("invalid prediction output");
    if (nativeKcTensor.dtype !== "float32" || nativeKcTensor.size !== KC_ELEMENTS || nativeKvTensor.dtype !== "float32" || nativeKvTensor.size !== KV_ELEMENTS) throw new Error("invalid graph state outputs");
    const predictionValues = prediction.dataSync(), nativeKc = nativeKcTensor.dataSync(), nativeKv = nativeKvTensor.dataSync();
    if (!allFinite(predictionValues) || !allFinite(nativeKc) || !allFinite(nativeKv)) throw new Error("graph produced a nonfinite value");
    trace.prediction.set(predictionValues, step * 30); writeSelected(trace.nativeKc, nativeKc, step, WIDTH); writeSelected(trace.nativeKv, nativeKv, step, WIDTH, WIDTH);
    const kcEncoded = encodeBoundary(nativeKc, previousKcFeed, args.arm), kvEncoded = encodeBoundary(nativeKv, previousKvFeed, args.arm);
    writeSelected(trace.feedKc, kcEncoded.feed, step, WIDTH); writeSelected(trace.feedKv, kvEncoded.feed, step, WIDTH, WIDTH);
    writeSelected(trace.highKc, kcEncoded.high, step, WIDTH); writeSelected(trace.highKv, kvEncoded.high, step, WIDTH, WIDTH);
    if (args.arm === "bf16_pair_boundary") { writeSelected(trace.lowKc, kcEncoded.low, step, WIDTH); writeSelected(trace.lowKv, kvEncoded.low, step, WIDTH, WIDTH); }
    summaries.push({token: step, kc: kcEncoded.summary, kv: kvEncoded.summary});
    const oldKc = kc, oldKv = kv;
    if (args.arm === "native") { kc = nativeKcTensor; kv = nativeKvTensor; }
    else {
      kc = tf.tensor(kcEncoded.feed, [1, LAYERS, HEADS, WIDTH], "float32");
      kv = tf.tensor(kvEncoded.feed, [1, LAYERS, HEADS, WIDTH, WIDTH], "float32");
      nativeKcTensor.dispose(); nativeKvTensor.dispose();
    }
    previousKcFeed = kcEncoded.feed; previousKvFeed = kvEncoded.feed;
    oldKc.dispose(); oldKv.dispose(); prediction.dispose(); iTensor.dispose(); xTensor.dispose();
  }
  const loopElapsedSeconds = Number(process.hrtime.bigint() - loopBegan) / 1e9;
  const tfMemoryAfter = tf.memory(); kc.dispose(); kv.dispose(); model.dispose();

  const selected = SELECTED_LAYERS.length;
  const arrays = {
    prediction_parameters: {values: trace.prediction, shape: [args.steps, 1, 30]},
    native_kc_state: {values: trace.nativeKc, shape: [args.steps, 1, selected, 1, WIDTH]},
    native_kv_state: {values: trace.nativeKv, shape: [args.steps, 1, selected, 1, WIDTH, WIDTH]},
    feed_kc_state: {values: trace.feedKc, shape: [args.steps, 1, selected, 1, WIDTH]},
    feed_kv_state: {values: trace.feedKv, shape: [args.steps, 1, selected, 1, WIDTH, WIDTH]},
    stored_high_kc: {values: trace.highKc, shape: [args.steps, 1, selected, 1, WIDTH]},
    stored_high_kv: {values: trace.highKv, shape: [args.steps, 1, selected, 1, WIDTH, WIDTH]},
  };
  if (args.arm === "bf16_pair_boundary") {
    arrays.stored_low_kc = {values: trace.lowKc, shape: [args.steps, 1, selected, 1, WIDTH]};
    arrays.stored_low_kv = {values: trace.lowKv, shape: [args.steps, 1, selected, 1, WIDTH, WIDTH]};
  }
  for (const [name, record] of Object.entries(arrays)) {
    const expected = record.shape.reduce((product, value) => product * value, 1);
    if (record.values.length !== expected || !allFinite(record.values)) throw new Error(`invalid ${name}`);
    writeFloat32(path.join(outputDir, `${name}.f32`), record.values);
  }
  const teacher = Buffer.allocUnsafe(args.steps * 4); for (let index = 0; index < args.steps; index++) teacher.writeInt32LE(pixels[index], index * 4);
  fs.writeFileSync(path.join(outputDir, "teacher_inputs.i32"), teacher);
  writeJson(path.join(outputDir, "rounding_summaries.json"), {arm: args.arm, summaries});
  const logicalBytesPerElement = args.arm === "bf16_boundary" ? 2 : 4;
  writeJson(path.join(outputDir, "trace.json"), {
    contract: "unmodified complete TFJS graph FP32 update/readout/prediction; storage boundary applied only before next token",
    arm: args.arm,
    armDefinition: args.arm === "native" ? "returned FP32 states feed the next graph call unchanged" : args.arm === "bf16_boundary" ? "returned states are RN-even BF16 rounded, expanded to FP32, then fed to the next call" : "returned x is stored as high=RN-even-BF16(x), low=RN-even-BF16(FP32(x-high)); FP32(high+low) feeds the next call; this is a residual-compensated two-word storage control, not compensated summation",
    source: "https://linear-transformers.com/ browser demo", licenseStatus: "unspecified for served weights; no redistribution claim",
    model: loaded.identity,
    input: {pixelsPath, pixelsSha256: sha256File(pixelsPath), batch: 1, steps: args.steps, startIndex: 0, positions: [0, args.steps - 1], ordering: "teacher input pixels[t] with graph position i=t from zero state"},
    runtime: {tfjs: tf.version.tfjs, converter: tf.version_converter, backend: tf.getBackend(), node: process.version, loopElapsedSeconds, tfMemoryBefore, tfMemoryAfter},
    selection: {layers: SELECTED_LAYERS, heads: [SELECTED_HEAD]},
    storage: {cacheElements: CACHE_ELEMENTS, logicalBytesPerElement, logicalCacheBytesBatch1: CACHE_ELEMENTS * logicalBytesPerElement, nativeFp32LogicalBytesBatch1: CACHE_ELEMENTS * 4, actualEmulation: "TFJS accepts only FP32 state tensors here; BF16 words are emulated in JavaScript and expanded/recomposed into FP32 tensors. Logical bytes are not actual allocation or RSS."},
    arrays: Object.fromEntries(Object.entries(arrays).map(([name, record]) => [name, {file: `${name}.f32`, dtype: "float32", byteOrder: "little-endian", shape: record.shape}])),
    teacherInputs: {file: "teacher_inputs.i32", dtype: "int32", byteOrder: "little-endian", shape: [args.steps, 1]},
    roundingSummaries: "rounding_summaries.json",
    randomSamplingExecuted: false,
  });
  return {loopElapsedSeconds, tfMemoryBefore, tfMemoryAfter};
}

async function main() {
  let args; try { args = parseArgs(process.argv.slice(2)); } catch (error) { console.error(`browser precision failed: ${error.message}`); return 1; }
  const outputDir = path.resolve(args.output); if (fs.existsSync(outputDir)) { console.error(`refusing existing output directory: ${outputDir}`); return 2; }
  const start = new Date(), began = process.hrtime.bigint(), memoryBefore = process.memoryUsage(); fs.mkdirSync(outputDir, {recursive: true});
  try {
    const runStats = await run(args, outputDir);
    const elapsed = Number(process.hrtime.bigint() - began) / 1e9, memoryAfter = process.memoryUsage(), maxRssKiB = process.resourceUsage().maxRSS;
    fs.copyFileSync(__filename, path.join(outputDir, "run_browser_precision.cjs"));
    writeJson(path.join(outputDir, "run.json"), {startUtc: start.toISOString(), endUtc: new Date().toISOString(), elapsedSeconds: elapsed, exitCode: 0, status: "success", command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)], node: process.version, loopElapsedSeconds: runStats.loopElapsedSeconds, rssBeforeBytes: memoryBefore.rss, rssAfterBytes: memoryAfter.rss, maxRssKiB, heapUsedAfterBytes: memoryAfter.heapUsed, tfMemoryBefore: runStats.tfMemoryBefore, tfMemoryAfter: runStats.tfMemoryAfter});
    fs.writeFileSync(path.join(outputDir, "run.log"), `start_utc=${start.toISOString()}\nend_utc=${new Date().toISOString()}\nelapsed_seconds=${elapsed.toFixed(9)}\nexit_code=0\nstatus=success\ncommand=${JSON.stringify([process.execPath, ...process.execArgv, ...process.argv.slice(1)])}\nnode=${process.version}\nrss_before_bytes=${memoryBefore.rss}\nrss_after_bytes=${memoryAfter.rss}\nmax_rss_kib=${maxRssKiB}\nheap_used_after_bytes=${memoryAfter.heapUsed}\n`, "utf8");
    writeJson(path.join(outputDir, "manifest.json"), {files: outputManifest(outputDir, new Set(["manifest.json", "complete.json"]))});
    writeJson(path.join(outputDir, "complete.json"), {status: "complete", manifestSha256: sha256File(path.join(outputDir, "manifest.json")), completedUtc: new Date().toISOString()});
    return 0;
  } catch (error) {
    const elapsed = Number(process.hrtime.bigint() - began) / 1e9;
    fs.writeFileSync(path.join(outputDir, "run.log"), `start_utc=${start.toISOString()}\nend_utc=${new Date().toISOString()}\nelapsed_seconds=${elapsed.toFixed(9)}\nexit_code=1\nstatus=failed\nerror=${error.name}: ${error.message}\nstack=${JSON.stringify(error.stack || null)}\n`, "utf8");
    writeJson(path.join(outputDir, "failure.json"), {status: "failed", errorType: error.name, error: error.message, stack: error.stack || null, endedUtc: new Date().toISOString()});
    console.error(error.stack || error.message); return 1;
  }
}

if (require.main === module) main().then((code) => { process.exitCode = code; });
module.exports = {roundFloat32ToBf16, roundFloat32ToBf16InPlace, encodeBoundary, CACHE_ELEMENTS, KC_ELEMENTS, KV_ELEMENTS};
