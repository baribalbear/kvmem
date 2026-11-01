# Radiance Windows / Linux AMD backend

Build and run with `tools/build.py --platform auto|windows|linux` and
`tools/serve.py --platform auto|windows|linux` on the target OS.

The adapter implements native page ownership, shared RAM/SSD spill and restore,
recurrent checkpoints, multiple retained conversations and concurrent prefix
branches. Query uses the common page-softmax or meanq-cosine scorer, bounded
replay, graph recall and LRFU. Prefill/decode Pressure selects resident history;
ordinary tool continuation does not recall cold history.

The shared console launcher supports one or two AMD GPUs, up to eight concurrent text/image
requests, fixed or native cost-adaptive MTP (up to seven drafts, default three), and process-local
SSD tiering. SSD files are temporary: a server restart does not restore sessions.
CPU storage work yields per request; failed optional cache parking drops that
entry, while a failed required read fails only the affected request. GPU transfer
failures remain fatal. RAM/SSD quotas are shared by active and retained sessions;
bounded I/O staging, statistics, model weights and runtime workspaces are separate.

Two GPUs use native tensor parallelism with exact collective transport, not a
layer split. Query heads and KV heads must divide evenly across the devices.
Each rank owns its KV/state bytes; accepted statistics are combined by head,
and a working set becomes executable only after every rank is ready. Prefix
branches, checkpoints and RAM/SSD quotas cover the complete set of rank shards.
The Linux dual-R9700 qualification includes a host-staged collective path with
peer-to-peer access disabled; it is not a throughput comparison with one GPU.

On systems where HSA cannot confirm CPU access to the VRAM pool, AQL kernel
argument rings and recorded tapes use the existing coherent host allocation.
Model weights and KV payloads remain on their selected GPU devices.

The native worker also rejects draft windows above seven before loading a model.
This keeps the deployment within the existing fused GDN kernel's eight-position
capacity (one target token plus up to seven drafts); no native kernel is changed.

Integration testing uses Linux, Radeon AI PRO R9700 (gfx1201), Qwen3.8-27B MXFP4
and FP8 KV. Native still-image input requires a qwen35 container with vision
weights, an FFmpeg-enabled build, and a GPU-resident encoder. Enable `Vision`
in the shared launcher; `VisionTokens` limits image tokens after spatial merge.
Image content and native rotary layout participate in retained-prefix identity.
Complete image pages remain required through Query and Prefill/Decode Pressure.
KV/state reuse does not currently skip native image preprocessing/encoding.

BF16/FP8 native memory contracts are tested separately. Windows AMD,
other models/devices, configurable n-gram drafting and DFlash combinations
still require qualification; their shared-launcher gates remain explicit.
Video/audio input and CPU encoder placement are not enabled by this adapter.

Original standalone profile: RX9060XT/gfx1200, BUDGET36864, generation reserve16384,
batch512, context262144, FP8 K/V, MTP3. Service: localhost18380, model radiance-local.

- [Build and deployment](../../docs/radiance-integration.md)
- [Client examples](../../docs/radiance/CLIENT-QUICKSTART.md)
- [Performance results](../../docs/radiance/PERFORMANCE-COMPARISON.md)
- [WSL runtime preparation](wsl-runtime/README.md)

Pinned source and patch metadata are in upstream.json. Preserve upstream
LICENSE/NOTICE with redistributed sources and binaries.
