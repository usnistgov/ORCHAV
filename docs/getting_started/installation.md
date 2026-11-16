# Installation

Install ORCHAV from source to use the included scenarios, assets, and examples.

## Prerequisites

- Python 3.12 with its `venv` and `ensurepip` modules.
- A graphics driver for the desktop Visualizer.
- For ray tracing, a CUDA-capable NVIDIA GPU and compatible driver, or an LLVM
  CPU backend. Follow the [Sionna RT installation guide](https://nvlabs.github.io/sionna/installation.html).

For playback, frame inspection, and reprocessing saved observations, you can
use a machine without a ray-tracing backend. Linux is the main development
and test platform. For the workflows tested on Windows and Apple Silicon
macOS, see [platform support](../reference/compatibility.md#platform-support).

For CPU generation, follow the [Linux LLVM setup](#linux-cpu-ray-tracing)
or [Apple Silicon setup](#apple-silicon-macos) after installing ORCHAV.

## Recommended Source Install

Get the source for your release from [usnistgov/ORCHAV](https://github.com/usnistgov/ORCHAV)
and open a terminal in the checkout.

### 1. Create an environment

On Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows, use a short local path such as `C:\src\ORCHAV`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the Windows `py` launcher is unavailable, use `python` with a 64-bit
Python 3.12 installation. If PowerShell blocks activation, see
[activation troubleshooting](../help/troubleshooting.md#powershell-blocks-virtual-environment-activation).

If environment creation reports that `ensurepip` is unavailable, install the
virtual-environment package for the same Python version. On Debian or Ubuntu,
this is usually `python3.12-venv`; availability depends on the Python package
source. On managed systems, use a complete Python 3.12 distribution supplied by
your administrator. Then repeat environment creation before installing ORCHAV.

### 2. Install ORCHAV

With the environment active:

```text
python -m pip install --upgrade pip
```

Install the checkout:

```text
python -m pip install -e .
```

The editable installation uses the code in your checkout. Keep this environment
active when you run the Generator, Visualizer, or command-line tools.

### 3. Check the installation

Continue with [Verify Installation](#verify-installation), then run the
[Quickstart](quickstart.md).

### Linux CPU Ray Tracing

After installing ORCHAV, select an LLVM shared library supplied by your system
or environment. This example uses LLVM 18. Replace its path with your compatible
LLVM library:

```bash
export CUDA_VISIBLE_DEVICES=""
export DRJIT_LIBLLVM_PATH=/path/to/llvm18/lib/libLLVM-18.so.1
python - <<'PYCODE'
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import drjit as dr
import sionna.rt
result = dr.maximum(mi.Float([-1, 2, 3]), 0)
dr.eval(result)
print(mi.variant(), result)
PYCODE
```

Expect `llvm_ad_mono_polarized` and values `[0, 2, 3]`. This compiles a small
arithmetic operation; an import alone does not check compilation. Keep the
variables set in the terminal used for CPU generation. CPU tracing can be
slower than CUDA, so start with the [Quickstart](quickstart.md).
For failures, see [backend troubleshooting](../help/troubleshooting.md#mitsuba-and-drjit-backend-messages).

### Linux Interactive Visualization

On Debian or Ubuntu, Qt may need the XCB cursor library:

```bash
sudo apt install libxcb-cursor0
```

For other display errors, see [Visualizer startup troubleshooting](../help/troubleshooting.md#visualizer-does-not-start-qt-or-display-errors).

### Supplied Wheel Installation

If you received a wheel, install it in an active Python 3.12 environment:

```text
python -m pip install "path/to/orchav-0.2.0-py3-none-any.whl"
```

Run asset-based workflows from the matching source checkout, which supplies
scenarios, scene and target assets, lighting resources, and repository configuration.

## Apple Silicon macOS

Use a native `arm64` Python 3.12 installation and the default `pygfx` renderer.
Ray tracing uses LLVM on the CPU. With [Homebrew](https://brew.sh/):

```bash
brew install llvm
```

If Dr.Jit cannot find LLVM, set its library path before launching ORCHAV:

```bash
export DRJIT_LIBLLVM_PATH="$(brew --prefix llvm)/lib/libLLVM.dylib"
```

On a managed Mac, use the LLVM installation provided by your administrator.
CPU generation can be much slower than CUDA, especially with large ray budgets.

If `import open3d` fails with
`Library not loaded: /opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib`, install
the system library and retry the import in your active Python environment:

```bash
brew install libusb
```

Verify the import:

```bash
python -c "import open3d; print(open3d.__version__)"
```

On a managed Mac, use the libusb dependency provided by your administrator.

## Verify Installation

Check that the Generator entry point is available:

```text
orchav-generator --help
```

Check the Visualizer entry point:

```text
orchav-visualizer --help
```

Check installed dependency compatibility:

```text
python -m pip check
```

For sensing, also check `orchav-sensing --help`. The included tutorials are in
the checkout under `scenarios/sensing/`.

For generation, also check the ray-tracing backend:

```text
python -c "import sionna.rt; import mitsuba as mi; print(mi.variant())"
```

Expect a name beginning with `cuda_` for NVIDIA or `llvm_` for CPU generation.
See [backend troubleshooting](../help/troubleshooting.md#mitsuba-and-drjit-backend-messages)
if initialization fails.

Run the [Quickstart](quickstart.md) to generate and view your first frames.
In PowerShell, enter multi-line Bash commands on one line or replace each
trailing `\` with a PowerShell backtick.

## Optional Extras

Install only the features you need:

| Feature | Install command |
|---|---|
| Live and remote streaming | `python -m pip install -e ".[grpc]"` |
| Jupyter notebooks | `python -m pip install -e ".[jupyter]"` |
| OpenStreetMap scenes | `python -m pip install -e ".[osm]"` |
| [Import AMASS motion with a SMPL model](../generator/motion/smpl_mesh_generation.md) | `python -m pip install -e ".[smpl]"` |
| Native RCS targets | `python -m pip install -e ".[native-rcs]"` |
| Development tools | `python -m pip install -e ".[dev,grpc]"` |

Extras can be combined, for example `python -m pip install -e ".[grpc,jupyter]"`.
See [Contributing](../../CONTRIBUTING.md#checks) for development checks.

### Native RCS Targets

For Sionna RT's constant-RCS and TR 38.901 target models, install the
`native-rcs` extra. It selects Sionna RT 2.2.0. Mesh/material sensing works
with the base installation.

Keep the Mitsuba and Dr.Jit versions selected by Sionna RT together.
See [Native RCS Targets](../sensing/target_models.md) for examples
and model limits.

### OpenStreetMap (OSM) Scenarios

Install the `osm` extra before running a geographic preparation script or a
scenario with `scene.source: osm`. The first run may download map data and
other geographic inputs. Viewing an already generated scene does not require
those downloads.

See [OpenStreetMap Scene Generation](../generator/geography/osm_scene_generation.md)
for configuration and caching.

---

Home: [Documentation](../README.md) | Next: [Quickstart](quickstart.md)
