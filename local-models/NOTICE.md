# Qwen companion payloads

Model Gateway local AI serves `mlx-community/Qwen3.8-27B-8bit` and
`mlx-community/Qwen3.8-27B-4bit`, conversions of `Qwen/Qwen3.8-27B`. The
upstream model cards declare Apache-2.0; the license text is supplied as
`LICENSE-Qwen-Apache-2.0.txt`. Retain the upstream README and this attribution
when distributing a companion payload.

Each manifest pins a revision and records every file's byte count and SHA-256,
checked against the upstream revision's Git blob IDs and LFS SHA-256 values
(`python -m src.hf_manifest REPO REVISION MODEL_ID` rebuilds one). No weight
modifications or re-quantization were performed.

| Manifest | Revision |
|---|---|
| `qwen3.8-27b-8bit.json` | https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/tree/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 |
| `qwen3.8-27b-4bit.json` | https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/10c35caafbb80f7dc6a7a432cdd11af10a6d4818 |

Weights are downloaded on request (`model-gateway local-ai add`), not Git
content. `src/model_payload.py` proves integrity against this release manifest;
signing and a trusted distribution channel remain necessary for publisher
authenticity.
