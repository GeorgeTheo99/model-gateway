# ADR 0004: Local Image and Video Generation

- **Status:** Accepted; implemented in 0.9.0 (media-gen 0.1.0)
- **Date:** 2026-10-28

## Context

Model Gateway routes chat models only: three chat protocols, a token ledger,
and a catalog every consumer (Pi, my-ai, Home Assistant) reads as a list of
chat models. A Mac with enough unified memory can also generate images and
video with open models (FLUX 3 Image Klein, Qwen-Image 2.1, LTX-2.5) through MLX, but:

- Generation is not chat: binary outputs, file storage, and video jobs that
  run for minutes. A video cannot be one HTTP request.
- The engines (mflux, ltx-2-mlx, torch) are several GB, move
  weekly, and a diffusion model that runs out of memory must not take chat
  routing down.
- oMLX (0.7) has no image or video endpoints.
- Image models and oMLX share unified memory; macOS kills a process rather
  than swap a GPU working set.

## Decision

### 1. A separate service: media-gen

Generation runs in **media-gen** (its own repository), a loopback service with
OpenAI-compatible endpoints:

- `POST /v1/images/generations`: synchronous, `b64_json` only.
- `POST /v1/videos` (JSON, or multipart with an `input_reference` image),
  `GET /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`,
  `DELETE /v1/videos/{id}`: the OpenAI Videos job shape.

One job runs at a time, in a spawned worker process that holds one model, so
unloading or switching a model frees all of its memory, a cancelled job stops
at once, and a crash fails only that job. Before a load, media-gen checks free
memory and unloads oMLX's least recently used idle models (never pinned,
loading, or recently used ones) until the model fits; otherwise it answers
`503 insufficient_memory` with `Retry-After`. Models are curated: each is a
set of Hugging Face repositories pinned at a commit, with every file's size
and hash, downloaded HTTPS-only, resumable, rate-limited and verified.

### 2. The gateway is the front door

The gateway routes the same endpoints to a `media` provider (default
`http://127.0.0.1:9112/v1`), with its own authentication, model allowlists and
ledger:

- **Modality.** A catalog row may declare `modality: image` or `video`
  (absent means `chat`). Chat endpoints refuse non-chat models, media
  endpoints refuse every other modality.
- **Discovery is opt-in.** Non-chat rows are left out of `/v1/models`,
  `/v1/models/canonical`, `/v1/debug/thinking`, the federation export and the
  profile route inventory, because every current consumer reads those as chat
  models (my-ai also rejects unknown row keys). `?include=modality` on
  `/v1/models` or `/v1/models/canonical` adds them, each row with `modality`.
- **No retries or fallbacks.** Generation is expensive and `POST /v1/videos`
  is not idempotent, so the gateway sends each request once. Profiles,
  model fallbacks, pools and federation do not apply to media; selectors are
  refused.
- **Video ownership.** A video belongs to the credential that created it. The
  gateway records each video id with its owner and answers `404` for any
  other caller, so one consumer can neither read nor list another's videos.
- **Ledger.** `POST /v1/images/generations` and `POST /v1/videos` are
  recorded (no tokens; local media rows are `unmetered`, so cost 0). Reads are
  not.

### 3. Optional, gateway-managed install (phase 2)

media-gen is not part of the gateway package. `model-gateway media add
<model>` installs it the way local AI installs oMLX: a locked project
(`media-runtime/`) built with uv, its own LaunchAgent (`com.local.media-gen`,
loopback only), a generated API key, the pinned model download, the `media`
provider and the curated rows in the shipped catalog layer. Before wiring the
gateway it checks that media-gen serves the model, and an image model must
generate one image.

### 4. Owner-supplied adapters stay local (0.14.0)

The curated models remain the shipped defaults. An owner may add a LoRA profile
on a curated model that accepts adapters (`model-gateway media custom`): checked
for compatibility at import, kept in media-gen's home outside any release,
disabled until enabled, routed under its own `custom-NAME` name with
`custom_base`, and never substituted for the curated model.
[deployment.md](../deployment.md#custom-lora-profiles) has the details.

## Consequences

- Clients that never ask for `modality` see no change.
- The media provider joins oMLX as a local provider for pricing (`unmetered`).
- Video outputs live on the media-gen host and expire (24 hours by default).
- Image-to-image, edits and federation of media are out of scope.
