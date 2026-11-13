# Windows release in CI

`.github/workflows/windows-release.yml` builds the kvmem Windows x64 release ZIP: the llama.cpp and
NInfer workers with their runtime DLLs, the offline console (embeddable Python 3.12.10 and psutil),
and the unified chat UI, packaged by `scripts/windows/package-backends.ps1 -Backend all`.
PolyStrata is not included.

It runs only when started manually: **Actions → KVMem Windows release → Run workflow**, with a
branch, a CUDA version and an architecture list. The defaults are CUDA 13.4.3 and `120a` (RTX 50
series). A `120a` NInfer build uses `NINFER_SM120_NATIVE=ON`, as `build-backends.ps1` does. The
NInfer startup log names the build as `<version>+kvmem-<commit>-ninfer-<patched tree>-sm<archs>`.

`<version>` comes from `scripts/release-version.py`: the newer of `VERSION` and the highest stable
kvmem/kvmem release tag (`vX.Y.Z`) the built commit contains. Upstream tags releases without always
bumping `VERSION` (`v0.19.1` still has `VERSION` 0.19.0), and a fork gets none of upstream's tags,
so the jobs check out the commit history (`fetch-depth: 0`, `filter: tree:0`) for this lookup.

| Job | Work |
|---|---|
| `dependencies` | Builds vcpkg's FFmpeg and curl for NInfer and caches them |
| `ninfer-objects` (4 shards) | Compiles NInfer's objects into a ccache |
| `llamacpp` | `scripts/windows/build.ps1 -BuildOnly` for `llama-kvmem-server.exe` |
| `ui` | `scripts/build-webui.py`, the lightweight chat the console requires |
| `package` | Links `ninfer-serve.exe`, packages and archives the release |

The CUDA install includes `nvml_dev` for `nvml.h`. Without that header, the workers compile out
their GPU monitor, and the console's GPU card shows no load, VRAM, temperature or power. The workers
load `nvml.dll` from the NVIDIA driver at run time, so no NVML file is packaged.

The artifact is `KVMem-<version>-windows-x64-cuda<CUDA>-sm<archs>`, containing the ZIP and its
SHA-256. Extract it and run `START-HERE-EN.bat` or `START-HERE-CN.bat`.

The runners have no GPU. The packager still checks every DLL dependency and starts each worker
with a clean `PATH` for `--help` and `--kvmem-capabilities`. To make this possible, the llama.cpp
worker delay-loads `nvcuda.dll`, and the packager never resolves or ships the driver DLL. No
inference or GPU tests run, so `provenance/validation.json` is absent. Validate the ZIP on a GPU
before publishing it.
