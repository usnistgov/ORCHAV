# Troubleshooting

Start with the symptom and follow the smallest relevant check.

```text
Windows installation reports a path that is too long -> use a shorter local checkout and environment path
Visualizer does not start -> Qt/display checks
Visualizer starts, no frames -> scenario.yaml and frame outputs
Live Generator fails -> running Generator, matching endpoint, and controlling client
Remote HDF5 fails -> completed frames, frame-file server, and matching endpoint
Local pygfx slows after changing displays -> adapter, window size, DPI, and power checks
Visualizer is slow over remote desktop -> presentation and remote-display checks
Large scenario is slow -> renderer, filters, cache, and ray-tracing quality
```

## PowerShell blocks virtual-environment activation

This is a PowerShell execution-policy setting, not an ORCHAV error. Allow
locally created scripts for the current PowerShell window, then activate the
environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

`-Scope Process` means the change ends when that PowerShell window closes. It
does not change the user or machine policy.

## Windows installation reports a path that is too long

Errors such as `WinError 206`, “filename or extension is too long,” or a file
that unexpectedly cannot be found during package installation can mean that
the checkout, virtual environment, and a dependency's internal directories
exceed a Windows tool's path limit.

Move or clone ORCHAV to a short writable local path such as
`C:\src\ORCHAV`, create a new Python 3.12 environment there or under a short
path such as `C:\venvs\orchav`, and retry the installation. Avoid deeply
nested folders and OneDrive-synchronized Desktop or Documents directories for
the checkout and environment. Enabling Windows long-path support may help some
tools, but it does not make every Python package or installer path-aware, so a
short local path is the most portable remedy.

If installation did not finish successfully, treat later import, test, or
launch errors as consequences of the incomplete environment. Recreate the
environment and confirm that installation completes before diagnosing an
ORCHAV runtime problem.

## Choose A Linux Display Setup

If you are sitting at the Linux machine or already using its normal desktop
session, launch the Visualizer normally. You do not need VNC, SSH X11
forwarding, or a manual `DISPLAY` override.

If you are remote and need interactive 3D, a VNC or other remote desktop
session is usually the best starting point. Open a terminal inside that
session, confirm its assigned display with `echo "$DISPLAY"`, and launch ORCHAV
from the same terminal. VNC keeps the application and display server on the
host and sends compressed screen updates. SSH X11 forwarding can be more
sensitive to network round trips and graphics-extension support.

For automation that does not need an interactive window, use
[Headless Rendering](../visualizer/headless_rendering.md).

## Choose A macOS Display Setup

On a Mac's normal local desktop, launch the Visualizer normally. No `DISPLAY`
setting, X11 forwarding, or remote-display setup is needed. macOS v0.1 uses the
default pygfx renderer, which reaches Apple's native Metal graphics API through
wgpu.

For interactive use from another machine, start with macOS Screen Sharing or
another full remote-desktop session, then launch ORCHAV from a terminal inside
that desktop. This keeps the Qt application and graphics context on the Mac.
SSH X11 forwarding is not the normal path for a Qt/Metal application and can
add latency or graphics-backend complications. Use
[Headless Rendering](../visualizer/headless_rendering.md) when no interactive
window is needed.

## Visualizer does not start (Qt or display errors)

- Ensure PySide6 is installed in the active environment.
- On a remote Linux host, choose an interactive remote desktop or headless
  execution as described above.

If Qt reports that `xcb-cursor0` or `libxcb-cursor0` is needed, or that the
`xcb` platform plugin was found but could not be loaded, the Python package is
installed but an operating-system runtime is missing. On Debian or Ubuntu,
install it and retry the same Visualizer command:

```bash
sudo apt install libxcb-cursor0
```

On another distribution, install its equivalent XCB cursor runtime. ORCHAV
does not install operating-system display libraries.

On Apple Silicon, an abort saying that Qt requires the `neon` processor feature
usually indicates an interpreter, wheel, or process-environment mismatch rather
than a normal ORCHAV error. Confirm that Python is running natively as `arm64`
and record the installed Qt bindings:

```bash
uname -m
file .venv/bin/python
.venv/bin/python -c 'import platform; print(platform.machine())'
.venv/bin/python -m pip show PySide6 PySide6-Essentials shiboken6
```

If the architectures do not agree, recreate the environment with a native
Apple Silicon Python 3.12 interpreter and reinstall ORCHAV. Also check for
custom `QT_` environment variables before treating the message as a PySide6 or
Qt compatibility defect.

## macOS Python cannot import Expat

Before installing ORCHAV with a selected macOS Python 3.12 interpreter, run:

```bash
python3.12 -c 'import platform, xml.parsers.expat; print(platform.machine(), "Expat OK")'
```

If this fails, including with a missing-symbol error from `pyexpat`, the
selected Python and Expat installation is internally incompatible. ORCHAV does
not supply or repair Python's Expat extension. Repair or replace the
interpreter, confirm native `arm64` execution, and rerun the preflight before
installing ORCHAV.

## Open3D import errors

- Verify that the Python environment matches the installed Open3D wheel.
- If you upgraded Python, reinstall Open3D in the new environment.

These checks concern importing the package. The Open3D/Filament Visualizer
renderer is unavailable on macOS in v0.1; an explicit request is rejected
before Qt starts. Use `--renderer pygfx` on macOS. ORCHAV still installs Open3D
there for geometry utilities used outside the renderer.

## pygfx / wgpu errors

- **"No GPU adapter found"** or **"RuntimeError: No wgpu adapter"**: wgpu
  requires a compatible native graphics backend. Update the graphics driver
  and inspect the adapter with
  `python -c "import wgpu; print(wgpu.gpu.request_adapter_sync().info)"`.
- **Blank window or no rendering**: Ensure the active environment has the
  normal ORCHAV runtime dependencies installed (`python -m pip install -e .`).
  Headless, VNC, and virtual-display environments can select a hardware or
  software adapter depending on their graphics setup. Inspect it with
  `python -c "import wgpu; print(wgpu.gpu.request_adapter_sync().info)"`.
- **Import errors for pygfx**: Reinstall ORCHAV in the active environment and
  verify the renderer packages with `python -m pip show pygfx wgpu`.

Messages such as `Unable to find extension: VK_EXT_physical_device_drm`, or
warnings about DRI3 or EGL, can appear in VNC or headless sessions. They do not
by themselves establish success or failure. Check the exit status and confirm
that the expected window or image was produced. Adapter enumeration lists
available adapters, but it does not prove which adapter presented a particular
window. Investigate further when output is missing, the command fails, or
performance suggests an unexpected software fallback.

## No frames found

- For the normal `files` workflow, confirm that the scenario directory contains
  `frames/frames_manifest.json` and its generated HDF5 chunks.
- If `scenario.yaml` has a read-only `data.files.directory` or `pattern`
  override, confirm that it selects an existing frame set. These fields do not
  redirect normal output from the Generator.
- Confirm that the Generator completed successfully.

## Live Generator (gRPC) connection fails

Live Generator mode connects the Visualizer to a separately running Generator
and requests frames on demand. It is different from Remote HDF5, which serves
an existing frame set.

- Start the Generator before the Visualizer, and select `live_grpc` for both
  processes.
- Verify that the client endpoint and Generator listener use the same port.
  The normal Live Generator port is `50051`. Use `--grpc-port` on both commands
  when overriding it.
- Do not connect a `live_grpc` Visualizer to the Remote HDF5 frame-file server.
  They are different servers and are not interchangeable.
- Only one Visualizer can control a Live Generator session. If startup reports
  `RESOURCE_EXHAUSTED` or an active controlling client, close the previous
  Visualizer before reconnecting.
- The Generator listens on `127.0.0.1` by default. For another machine, prefer
  an SSH tunnel or explicitly bind the server to an interface on a trusted
  private network and allow the selected port through the firewall.
- A disconnected session is not resumed automatically. Make the Generator
  available, then reopen the scenario.

See [Live Generator (gRPC)](../../scenarios/visualizer/data_modes/live_grpc/README.md)
for the complete two-terminal command sequence.

## Remote HDF5 connection fails

Remote HDF5 reads a completed, persisted frame set through the frame-file
server. It does not run the Generator or recompute paths.

- Generate the scenario in `files` mode first and confirm that its `frames/`
  directory contains `frames_manifest.json` and the generated HDF5 chunks.
- Start `python -m generator.io.grpc.file_server` with `--frames-dir` pointing
  to the scenario directory that contains that `frames/` child.
- Select `remote_hdf5` in the Visualizer, and verify that its endpoint uses the
  frame-file server's port. The included example uses port `50052`.
- Do not point a `remote_hdf5` Visualizer at a Live Generator service.
- `--bind-host` changes the server listener, while the Visualizer's
  `--grpc-port` changes only its client port. Configure the remote host in the
  scenario or preserve its localhost endpoint with an SSH tunnel.

See [Remote HDF5 Playback](../../scenarios/visualizer/data_modes/remote_hdf5/README.md)
for the generation, serving, and connection commands.

## Mitsuba and Dr.Jit backend messages

A backend check or Generator run may show messages from Mitsuba and Dr.Jit.
Interpret each message together with the result of the command.

| Message begins with | What it means | When it is acceptable |
|---------------------|---------------|-----------------------|
| `jitc_llvm_init()` | Dr.Jit could not load a usable LLVM API for CPU execution. | The intended backend is CUDA and the command exits successfully. |
| `jit_registry_shutdown()` | Dr.Jit found Mitsuba objects still registered while Python was closing. | The command exits successfully and writes the expected output. |
| `jit_malloc_shutdown()` | Dr.Jit found memory still allocated while Python was closing. | The command exits successfully and writes the expected output. |

For example, the messages can look like this:

```
jitc_llvm_init(): LLVM API initialization failed ..
jit_registry_shutdown(): leaking 8 instances of type "mitsuba::BSDF".
jit_malloc_shutdown(): leaked
```

First check the command outcome. Treat the run as a failure if it ends with an
exception, returns a nonzero exit status, or does not produce the expected
output. After generation, use `orchav-inspect` as documented by the scenario to
confirm that a frame can be loaded.

Then confirm which backend is active:

```bash
python -c "import sionna.rt; import mitsuba as mi; print(mi.variant())"
```

- A variant beginning with `cuda_` means that
  [Sionna RT](https://nvlabs.github.io/sionna/) is using the GPU. In
  that case, an LLVM initialization message concerns the unused CPU backend.
  It does not indicate a CUDA failure when generation succeeds.
- A variant beginning with `llvm_` means that Sionna RT is using the CPU. If
  LLVM cannot initialize, CPU generation cannot proceed. Follow the
  [Apple Silicon setup](../getting_started/installation.md#apple-silicon-macos)
  on macOS or install LLVM through the platform package manager.
- If importing Sionna RT fails before a variant is reported, repair the intended
  backend rather than ignoring the initialization message.

The LLVM message commonly means that LLVM is missing, cannot be found, or is
incompatible with Dr.Jit. It requires action for CPU execution but not for a
successful CUDA run.

The shutdown messages can appear because Python releases Mitsuba and Dr.Jit
objects in an order that triggers cleanup warnings. After a successful command,
no action is required. If they follow an exception or missing output,
investigate the earlier error instead of treating the shutdown messages as the
cause.

## Local pygfx playback becomes slow after changing monitors or GPUs

Hybrid-GPU laptops can route the built-in panel and external display ports
through different adapters. The pygfx/wgpu adapter is selected when the
renderer starts and does not change when a window is moved to another monitor.
Close the Visualizer before testing another adapter.

First restore the window instead of maximizing it. Compare the same scenario
at the same physical renderer size: a high-DPI or 4K display can make a larger
window substantially more expensive even when target and MPC processing are
unchanged. Also confirm the machine is connected to AC power and uses the
intended Windows or system power mode. ORCHAV's `--max-performance` option is a
separate cache policy and does not change the system power mode.

Then compare ordinary wgpu selection with an explicit adapter:

```bash
python -c "import wgpu; print([a.info['device'] for a in wgpu.gpu.enumerate_adapters_sync()])"
```

```powershell
# Remove a selection inherited from the current shell.
$env:PYGFX_WGPU_ADAPTER_NAME = $null

orchav-visualizer --renderer pygfx `
  --pygfx-present-method screen `
  --scenario path/to/scenario

orchav-visualizer --renderer pygfx `
  --pygfx-present-method screen `
  --pygfx-adapter-name "ADAPTER NAME FROM THE LIST ABOVE" `
  --scenario path/to/scenario
```

Replace the example adapter name with one reported by the local wgpu runtime.
The CLI selection is process-local and overrides an existing
`PYGFX_WGPU_ADAPTER_NAME` value for that launch. It affects only pygfx/wgpu.
It does not change Sionna RT, CUDA, Mitsuba, Open3D, or generation.

Do not assume that the discrete GPU is always faster. Display routing, driver
backend, power policy, and cross-adapter transfer can change the result. Use
the same window size and presentation method for each comparison. The
`--pygfx-adapter-name` option requires `--renderer pygfx`.

## Visualizer playback is slow over VNC or another remote desktop

Remote presentation can dominate playback even when rendering uses a fast GPU.
The pygfx presentation methods have different tradeoffs:

| Method | Behavior | Typical use |
|--------|----------|-------------|
| `screen` | Presents through a direct GPU surface without full-frame readback. | Normal local Windows desktop and the first comparison for Windows RDP. |
| `bitmap` | Reads the rendered image back for Qt bitmap composition. | Compatibility path for Linux VNC and display setups without a usable direct surface. |
| `auto` | Delegates selection to rendercanvas. With the tested Qt stack, this normally resolves to `bitmap`. | Non-Windows default and an explicit compatibility comparison. |

Start with the method appropriate for the session:

| Session | First choice |
|---------|--------------|
| Local Windows desktop | Omit the option (ORCHAV selects `screen`). |
| Linux VNC | Omit the option (`auto`, normally `bitmap`) or force `bitmap`. |
| Windows Remote Desktop | Start with the inherited Windows `screen` default. Compare `bitmap` and `auto` if playback is slow. |
| Other local desktops | Omit the option (ORCHAV delegates to rendercanvas with `auto`). |

Force the Linux VNC-compatible path with:

```bash
orchav-visualizer --renderer pygfx \
  --pygfx-present-method bitmap \
  --scenario path/to/scenario
```

ORCHAV does not classify VNC, RDP, or other remote-session products when
choosing the Windows default. RDP application rendering and display delivery
depend on the host configuration, so compare methods with the same scenario and
window size instead of assuming Linux VNC behavior applies to RDP. Explicit
`auto` delegates to rendercanvas and therefore differs from omission on
Windows. If direct `screen` initialization fails, ORCHAV retries once with
`bitmap`. It does not change methods merely because measured playback is slow.

`--pygfx-present-method` does not affect Open3D. On Linux VNC, Open3D follows
the display's native GL path and may report Mesa llvmpipe. The v0.1 tests
exercise that as a functional path, but its timing is not NVIDIA
GPU-performance evidence. ORCHAV v0.1 runs Open3D directly; Open3D through
VirtualGL was not included in v0.1 testing. Use pygfx for GPU-backed VNC or
performance measurements, record the selected adapter, and do not interpret
software Open3D numbers as L40S or V100 performance. SSH X11 forwarding can
remain latency-bound.

The status-bar **Playback updates** value measures completed scenario-frame
pipelines on the server, not client-visible FPS. In **System -> Performance**,
**Render callback** is server-side callback cadence and update-to-callback
latency, while **Renderer submit** is time spent in the renderer call. These
server-side values can reflect time spent interacting with the virtual display,
but they do not measure VNC/RDP encoding, network delivery, or client redraw.

## Performance issues

- Large scenes benefit from GPU acceleration.
- Reduce ray tracing quality presets for faster generation.
- In the Visualizer, reduce displayed MPCs using filters.
- Increase the canonical cache for large scenarios:
  `export VIZ_CANON_CACHE_MB=4096`
- Use the pygfx renderer (`--renderer pygfx`) for scenarios with >100k MPCs.
- For repeatable renderer checks, use the
  [Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md).

---

Home: [Documentation](../README.md) | Related: [Installation](../getting_started/installation.md) | [Application Configuration](../reference/application_configuration.md)
