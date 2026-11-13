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

`vision_cpu_isa` (default `avx512`) sets the instruction set of NInfer's CPU Vision encoder, which
encodes images with `--vision-residency cpu`. Upstream compiles it for the compiler's baseline (SSE2 on
MSVC x64). The fork's `qwen-vision-cpu-isa` NInfer patch compiles that one file with `/arch:AVX512`
or `/arch:AVX2`, plus `/fp:contract` so MSVC fuses multiply-adds. With `avx512` the encoder's
projection tile also uses AVX-512 FMA intrinsics: the flags alone sped encoding up by only about 2%.
With CPU Vision the worker logs `CPU Vision encoder isa=avx512` (or `avx2`, `baseline`) at startup,
which shows the encoder was really compiled that way. An `avx512` build runs only on
CPUs with AVX-512, such as Ryzen 7000 and 9000; most Intel desktop CPUs lack it. Choose `avx2` for
any recent x64 CPU, or `baseline` for upstream's build. A non-baseline choice is added to the
build ID (`...-sm120a-cpuvision-avx512`) and to the artifact name. FMA rounds differently, so image
embeddings differ from a baseline build in the last bits.

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
