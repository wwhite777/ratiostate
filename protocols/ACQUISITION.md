# External model, MNIST and runtime identities

Saved-data replay needs no network, model or runtime acquisition. Fresh inference needs the external graph at `https://linear-transformers.com/models/linear/model.json` and its seven `group1-shardNof7.bin` files in the same directory. These assets are not in this repository. The exact expected identities, taken from the archived inference source manifest, are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `group1-shard1of7.bin` | 4194304 | `81e68fcfae9499fc116d8a85c87215f766f19c7d28d1c3b8ab2ab5e61650e965` |
| `group1-shard2of7.bin` | 4194304 | `b8589e74a115db7d1cea63cf6dcb776c87e64169cbc35133bbcee7912d5409c5` |
| `group1-shard3of7.bin` | 4194304 | `b0356e44d7cc033b8be2a92d1e7cb423850f1761b1741affe3f173db769b9ba7` |
| `group1-shard4of7.bin` | 4194304 | `d3ccdf02016baadeb3712a4fa9aec104c2854bc836dd7cd7861908b0da6c382d` |
| `group1-shard5of7.bin` | 4194304 | `59634fee8082996f4bc6977f30032a09ddf8fec371850b9064191500b5d3adf2` |
| `group1-shard6of7.bin` | 4194304 | `d5616b81909a30e96df27a179dca54c83fc20cd5ac1b4388b9e4ffee09074e3d` |
| `group1-shard7of7.bin` | 673276 | `bab7e2ca05ab24e858bd115b71db967abdf8ebf0f3514583644ccae3ad632b44` |
| `model.json` | 386110 | `304f1905c945d25b4848873feaa3ea04c1fa6c84fa9230b214c24d2b85b1f83e` |

The original test IDX files were `https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz` and `https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz`. The recorded SHA-256 values for the **uncompressed IDX** files are image `0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7` and label `ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2`. The repository includes only selected JSON pixel sequences, not the full MNIST test set. The original served model's training provenance and a standalone graph/weight redistribution grant are unknown; verify upstream access and terms before obtaining or redistributing these assets.

The tested graph runtime was Node 20.19.6, TensorFlow.js 2.0.0 CPU backend. `inference/.cache/tfjs_runtime/package.json` and `package-lock.json` record the dependency resolution, but `node_modules` is not vendored. An offline install was not newly tested for this public export. `inference/run_six.py --check-only` verifies explicit external model, pixel and runtime inputs before any optional inference. It is restricted to saved test image 6116, not a general retraining or new-sample evaluation tool.
