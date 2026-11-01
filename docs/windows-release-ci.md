# Windows release in CI

`.github/workflows/windows-release.yml` builds the kvmem Windows x64 release ZIP: the llama.cpp and
NInfer workers with their runtime DLLs, the offline console (embeddable Python 3.12.10 and psutil),
and the unified chat UI, packaged by `scripts/windows/package-backends.ps1 -Backend all`.
PolyStrata is not included.

It runs only when started manually: **Actions → KVMem Windows release → Run workflow**, with a
branch, a CUDA version and an architecture list. The defaults are CUDA 13.4.3 and `120a` (RTX 50
series). A `120a` NInfer build uses `NINFER_SM120_NATIVE=ON`, as `build-backends.ps1` does. The
NInfer startup log names the build as `<VERSION>+kvmem-<commit>-ninfer-<patched tree>-sm<archs>`.

| Job | Work |
|---|---|
| `dependencies` | Builds vcpkg's FFmpeg and curl for NInfer and caches them |
| `ninfer-objects` (4 shards) | Compiles NInfer's objects into a ccache |
| `llamacpp` | `scripts/windows/build.ps1 -BuildOnly` for `llama-kvmem-server.exe` |
| `ui` | `scripts/build-webui.py`, the lightweight chat the console requires |
| `package` | Links `ninfer-serve.exe`, packages and archives the release |

The artifact is `KVMem-<VERSION>-windows-x64-cuda<CUDA>-sm<archs>`, containing the ZIP and its
SHA-256. Extract it and run `START-HERE-EN.bat` or `START-HERE-CN.bat`.

The runners have no GPU. The packager still checks every DLL dependency and starts each worker
with a clean `PATH` for `--help` and `--kvmem-capabilities`. To make this possible, the llama.cpp
worker delay-loads `nvcuda.dll`, and the packager never resolves or ships the driver DLL. No
inference or GPU tests run, so `provenance/validation.json` is absent. Validate the ZIP on a GPU
before publishing it.
