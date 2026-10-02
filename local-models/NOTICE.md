# Qwen companion payload

Model Gateway local AI serves `mlx-community/Qwen3.8-27B-8bit`, a conversion
of `Qwen/Qwen3.8-27B`. The upstream model card declares Apache-2.0; the license
text is supplied as `LICENSE-Qwen-Apache-2.0.txt`. Retain the upstream README and
this attribution when distributing the companion payload.

`qwen3.8-27b-8bit.json` pins revision
`815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` and records every file's byte count and
SHA-256. These were checked against the upstream revision's Git blob IDs and
LFS SHA-256 values. No weight modifications or re-quantization were performed.

Source: https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/tree/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9

Weights are downloaded on request (`model-gateway local-ai add`), not Git
content. `src/model_payload.py` proves integrity against this release manifest;
signing and a trusted distribution channel remain necessary for publisher
authenticity.
