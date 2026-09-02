# Installation

ORCHAV v0.1 has been tested on the following configurations. The table records
the v0.1 validation performed on each system.

| Platform tested | Environment | v0.1 validation performed |
|---|---|---|
| Linux (primary release testing) | Ubuntu 22.04, four NVIDIA L40S GPUs, TigerVNC/XCB | Fresh source installs, all automated gates, all included scenarios and examples, every data mode, Scenario Builder, pygfx and Open3D desktop rendering, visual regression, and the renderer density trend pack |
| Linux (additional release testing) | Ubuntu 24.04, Tesla V100, TigerVNC/XCB | Fresh source installs, all automated gates and scenario validation, representative heavy generation and UI workflows, every data mode, notebook paths, Scenario Builder, both desktop renderers, and the renderer density trend pack |
| Windows (targeted testing) | Windows 11, native 64-bit CPython 3.12, NVIDIA RTX 2000 Ada-class and RTX 4070 Laptop GPUs, CUDA Sionna RT, local desktop | Across the two tested systems: source installation, configuration validation of all 27 included scenarios, ten representative CUDA producer workflows, examples, all three data modes (`files`, `online`, and `remote_file`), Scenario Builder, notebook and headless paths, and pygfx and Open3D desktop playback |
| macOS (Apple Silicon; targeted testing) | Apple M4, native-arm64 Python 3.12, LLVM CPU generation, pygfx/wgpu over Metal | Fresh source installs, targeted tests, configuration validation of all 27 included scenarios, 20 of 25 Sionna RT producer workflows with LLVM CPU generation, frame inspection, examples, notebooks, selected data modes, Scenario Builder, and pygfx desktop/notebook/headless rendering over Metal |

These are the configurations tested for v0.1, not an exhaustive list of systems
on which ORCHAV may run. Other configurations may work. Report reproducible
problems through [GitHub Issues](https://github.com/usnistgov/ORCHAV/issues)
with the operating system, hardware, Python version, driver versions, and the
smallest useful log.

## Prerequisites

- Python 3.12. ORCHAV targets Sionna and
  [Sionna RT](https://nvlabs.github.io/sionna/) 2.0.x.
- For workflows that generate or recompute ray-traced frames, use one of these
  Sionna RT backends:
  - **Primary NVIDIA configuration:** a CUDA-capable NVIDIA GPU and a driver
    compatible with Sionna RT 2.0.x. ORCHAV does not install the GPU driver.
    Follow the [Sionna installation guidance](https://nvlabs.github.io/sionna/installation.html)
    and verify below that Mitsuba reports a `cuda_` variant.
  - **Apple Silicon CPU configuration:** a native Apple Silicon
    Python 3.12 environment with the LLVM CPU backend described under
    [Apple Silicon macOS](#apple-silicon-macos).
- A graphics driver compatible with the selected Visualizer renderer.

Inspection, Python analysis, Local HDF5 Playback, and Remote HDF5 Playback of
an existing frame set do not invoke Sionna RT and do not require a CUDA or LLVM
ray-tracing backend. CPU generation has not received ORCHAV v0.1's complete
release-validation matrix.

For unlisted platforms, other GPU drivers, or very new NVIDIA GPU generations,
complete [Verify Installation](#verify-installation) and the
[Quickstart](quickstart.md) before regular use.

## Recommended Source Install

ORCHAV v0.1 is installed from a cloned source checkout. The commands below
separate the platform-independent steps from the shell-specific environment
activation commands. They use `.venv`, which is already ignored by Git. Choose
another directory if you prefer.

Plan for at least 20 GiB of free space on the volume holding the checkout,
virtual environment, package cache, and temporary installation files. This is
conservative installation guidance rather than a product minimum. Generated
frames, optional extras, and additional test environments require more space.

### 1. Clone The Repository -- All Platforms

```text
git clone https://github.com/usnistgov/ORCHAV.git
cd ORCHAV
```

On Windows, place the checkout and virtual environment in a short local path,
such as `C:\src\ORCHAV`, or another short directory that is writable without
administrator access. Avoid OneDrive-synchronized Desktop or Documents paths
and deeply nested working directories: scientific Python packages can contain
long internal paths, and some Windows tools still fail when the resulting path
exceeds their limit. See
[Windows path-length troubleshooting](../help/troubleshooting.md#windows-installation-reports-a-path-that-is-too-long)
if installation stops with this symptom.

### 2a. Create The Environment -- macOS And Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If `python3.12 -m venv` is unavailable or lacks `ensurepip`, use another native
Python 3.12 build that includes virtual-environment support.

On macOS, verify the selected interpreter before creating the environment:

```bash
python3.12 -c 'import platform, xml.parsers.expat; print(platform.machine(), "Expat OK")'
```

The architecture must be `arm64`, and the import must succeed. An import
failure, including a missing-symbol error from `pyexpat`, belongs to the
selected Python/Expat installation. Repair or replace that interpreter before
installing ORCHAV.

### 2b. Create The Environment -- Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The `py` launcher is convenient but is not required. If it is unavailable,
select a native 64-bit CPython 3.12 interpreter through `python`, verify it,
and create the environment with that same command:

```powershell
python -c "import platform, sys; assert sys.implementation.name == 'cpython'; assert sys.version_info[:2] == (3, 12); assert sys.maxsize > 2**32; assert platform.machine().lower() in ('amd64', 'x86_64'); print(sys.executable); print(sys.version); print(platform.machine())"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Do not continue if the verification command fails. A Conda-provided native
CPython 3.12 interpreter is suitable when it passes the same checks; ORCHAV
does not require the Python launcher or a particular Python distributor.

If PowerShell reports that scripts are disabled, use the focused
[activation troubleshooting](../help/troubleshooting.md#powershell-blocks-virtual-environment-activation).

### 3. Install ORCHAV -- All Platforms

With the environment activated, run:

```text
python -m pip install --upgrade pip
python -m pip install -e .
```

### Linux Interactive Visualization

pip installs ORCHAV's Python dependencies. A windowed Visualizer on Linux also
uses graphics and display libraries supplied by the operating system. On
Debian or Ubuntu systems using X11 or VNC, install the XCB cursor runtime if it
is not already available:

```bash
sudo apt install libxcb-cursor0
```

Package names differ on other distributions. Use the equivalent XCB cursor
runtime supplied by the operating system. See
[Visualizer does not start](../help/troubleshooting.md#visualizer-does-not-start-qt-or-display-errors)
if Qt finds the `xcb` platform plugin but cannot load it.

The audited Ubuntu virtual environment used approximately 8 GiB after
installation. Reserve additional space for the source checkout, package
caches, and generated scenario outputs. The exact total depends on the
selected extras and third-party GPU, ray-tracing, and visualization packages.

The remaining commands in this guide assume that the environment is active.
Editable installation keeps the package linked to the checkout, so local
changes are available without reinstalling.

Across the documentation, multi-line `bash` examples use `\` as the line
continuation. In PowerShell, enter the same command on one line or replace each
`\` with a PowerShell backtick.

Version 0.1 uses a source-checkout installation. The included scenarios, scene
and target assets, lighting resources, and repository configuration are not
embedded in the Python wheel, so run `python -m pip install -e .` from the
cloned repository. A wheel by itself does not provide the complete repository
assets, and v0.1 is not published to a Python package index.

ORCHAV currently targets Sionna and Sionna RT 2.0.x with Mitsuba 3.8.x. Newer
Sionna RT or Mitsuba minor versions should be tested with the checks below and
the first generated scenario before relaxing those dependency bounds.

## Apple Silicon macOS

The tested macOS configuration is an Apple M4 Mac mini with macOS 26.2 and a
native-arm64 Python 3.12 environment. Its producer workflows use
[Dr.Jit's LLVM CPU backend](https://drjit.readthedocs.io/en/stable/what.html),
while visualization uses pygfx/wgpu over Metal.

On macOS, wgpu uses Metal, Apple's native graphics API. The Open3D/Filament
Visualizer renderer is unavailable on macOS in v0.1; an explicit request is
rejected before Qt starts. Use the default `pygfx` renderer.
CPU generation may be substantially slower than the primary CUDA
configuration, and higher ray budgets can take several minutes.

Use a native Apple Silicon Python 3.12 environment. macOS's system Python is
not sufficient for this release. LLVM is an external prerequisite for CPU
generation. ORCHAV and pip do not install it. If you use
[Homebrew](https://brew.sh/), the following command assumes Homebrew is already
installed:

```bash
brew install llvm
```

Homebrew is shown here as one LLVM provider. It is not a required Python
provider, and a Homebrew Python installation with a broken Expat extension must
be repaired or replaced like any other incompatible interpreter.

If Dr.Jit does not discover Homebrew LLVM, point it to Homebrew's installed
shared library before launching ORCHAV:

```bash
export DRJIT_LIBLLVM_PATH="$(brew --prefix llvm)/lib/libLLVM.dylib"
```

On a managed Mac where Homebrew is unavailable, obtain LLVM and its
`libLLVM.dylib` path through your administrator. Custom relocation or
modification of LLVM libraries is outside the documented installation path.

## Verify Installation

Confirm that the command-line tools and intended Sionna RT backend are
available:

```text
orchav-generator --help
orchav-visualizer --help
orchav-validate --help
orchav-inspect --help
python -c "import sionna.rt; import mitsuba as mi; print(mi.variant())"
```

The final command should report a variant beginning with `cuda_` for the
primary NVIDIA configuration or `llvm_` for an explicitly configured CPU
environment. See [Mitsuba and DrJit backend messages](../help/troubleshooting.md#mitsuba-and-drjit-backend-messages)
if the intended backend does not initialize.

If a console script is unavailable, use its equivalent module command:

```text
python -m generator --help
python -m visualizer --help
python -m shared.cli.validate --help
python -m shared.cli.inspect --help
```

Continue with the [Quickstart](quickstart.md) for the first end-to-end frame
generation and visualization check.

## Optional Extras

The base install is sufficient for normal local generation and visualization.
Add only the optional feature groups needed by your workflow:

```text
python -m pip install -e .                     # runtime package with default pygfx renderer
python -m pip install -e ".[grpc]"             # gRPC streaming
python -m pip install -e ".[jupyter]"          # notebook display support
python -m pip install -e ".[dev]"              # tests, formatting, docs tools
python -m pip install -e ".[dev,grpc]"         # contributor smoke-test dependencies
python -m pip install -e ".[grpc,jupyter]"     # combined streaming + notebook install
```

The default install includes both desktop renderer packages. The Visualizer
offers pygfx and Open3D/Filament on Windows and Linux. On macOS, use pygfx.
The Open3D package remains installed because ORCHAV also uses its geometry
utilities outside the renderer. Notebook and standalone headless rendering use
pygfx. On Linux VNC, Open3D follows the display's native GL path and may use
Mesa llvmpipe. Run Open3D directly; Open3D through VirtualGL was not included
in v0.1 testing. Use pygfx for GPU-backed VNC or performance measurements,
record the selected adapter, and do not interpret software-rendered Open3D
results as L40S or V100 performance. The `jupyter` extra adds JupyterLab and
in-cell widget support used by the ORCHAV notebook examples. See [Notebook
Visualization](../visualizer/notebooks.md) for the Jupyter workflow. The
`grpc` extra contains the runtime libraries for live and remote frame
transport. The protobuf compiler is a contributor tool supplied by the `dev`
extra. Users of the checked-in protocol modules do not need it.

Contributors should install the development extras and follow the exact
quality, test, and generated-code checks in
[Contributing](../../CONTRIBUTING.md#checks).

---

Home: [Documentation](../README.md) | Next: [Quickstart](quickstart.md)
