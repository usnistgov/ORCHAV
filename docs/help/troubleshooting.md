# Troubleshooting

Choose the problem you are seeing:

- [PowerShell blocks environment activation](#powershell-blocks-virtual-environment-activation)
- [Windows installation path is too long](#windows-installation-reports-a-path-that-is-too-long)
- [Windows generation cannot open an existing mesh](#windows-generation-cannot-open-an-existing-mesh)
- [Visualizer does not start](#visualizer-does-not-start-qt-or-display-errors)
- [Visualizer opens but shows no frames](#no-frames-found)
- [Live Generator connection fails](#live-generator-grpc-connection-fails)
- [Remote HDF5 connection fails](#remote-hdf5-connection-fails)
- [Local preview reports a source mismatch](#local-what-if-preview-reports-a-source-identity-mismatch)
- [Playback slows after changing monitors or GPUs](#local-pygfx-playback-becomes-slow-after-changing-monitors-or-gpus)
- [Playback is slow over remote desktop](#visualizer-playback-is-slow-over-vnc-or-another-remote-desktop)
- [Large scenarios are slow](#performance-issues)
- [Ray-tracing backend initialization or shutdown messages](#mitsuba-and-drjit-backend-messages)
- [macOS Python cannot import Expat](#macos-python-cannot-import-expat)
- [Open3D import errors](#open3d-import-errors)
- [pygfx / wgpu errors](#pygfx--wgpu-errors)
- [OSM canopy tiles fail to download](#osm-vegetation-eth-canopy-tile-fails-to-download)
- [OSM water differs from WorldCover](#osm-vegetation-worldcover-reports-water-not-in-osm)
- [OSM generation is slow or Overpass times out](#osm-scene-generation-is-slow-or-overpass-times-out)

## PowerShell blocks virtual-environment activation

This is a PowerShell execution-policy setting, not an ORCHAV error. Allow
locally created scripts for the current PowerShell window, then activate the
environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Activate the environment:

```powershell
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

## Windows generation cannot open an existing mesh

Local PLY scene loading has a known Windows path limitation with the tested
[Sionna RT](https://nvlabs.github.io/sionna/) 2.2.0 and Mitsuba 3.9.1 stack. A
non-ASCII character in a parent directory, such as `Scene café`, can cause
Mitsuba's PLY `FileStream` to report "No such file or directory" even though
the mesh exists. The same scene and mesh load successfully from an ASCII-only
path.

Keep the scenario folder and referenced mesh assets under **ASCII-only full
paths**, for example `C:\orchav-runs\Scene_cafe`. Check every parent directory,
including a Windows user-profile directory with an accented name. Renaming
only the mesh file does not remove those characters. Copy the complete
scenario and asset tree to preserve relative references, update any absolute
asset references, then retry generation.

Saving, previewing, validating, or reopening the scenario in Scenario Builder
does not establish that Sionna RT can load its meshes for generation. Unicode
mesh-path generation remains unsupported on this tested Windows stack.
Long-path support is not a verified remedy for this failure.

## Choose A Linux Display Setup

On a normal local desktop, launch the Visualizer without changing `DISPLAY`.
For remote interactive use, launch it from a terminal inside the remote-desktop
session so Qt and the renderer use that session's display. SSH X11 forwarding
can be sensitive to network latency and graphics-extension support.

Use [Headless Rendering](../visualizer/headless_rendering.md) for automation
without an interactive window.

## Choose A macOS Display Setup

On the local desktop, use the default pygfx renderer without a `DISPLAY`
override. It uses Apple's Metal graphics API through wgpu. For remote use,
launch from a terminal inside macOS Screen Sharing or another full desktop
session. SSH X11 forwarding is not the normal display path for Qt/Metal.

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
and record the installed Qt bindings.

Check the operating-system architecture:

```bash
uname -m
```

Check the virtual environment's executable:

```bash
file .venv/bin/python
```

Check the architecture Python reports:

```bash
.venv/bin/python -c 'import platform; print(platform.machine())'
```

Record the installed Qt bindings:

```bash
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
renderer is unavailable on macOS. An explicit request is rejected
before Qt starts. Use the default pygfx renderer on macOS. ORCHAV still installs Open3D
there for geometry utilities used outside the renderer.

## pygfx / wgpu errors

For **No GPU adapter found** or a blank renderer, inspect the adapter available
to the active environment:

```bash
python -c "import wgpu; print(wgpu.gpu.request_adapter_sync().info)"
```

If no compatible adapter is available, check the native graphics driver and
session setup. VNC and headless environments can select software rendering.
For import errors, confirm the environment has the normal ORCHAV installation
and inspect its renderer packages:

```bash
python -m pip show pygfx wgpu
```

Warnings about `VK_EXT_physical_device_drm`, DRI3 or EGL do not by themselves
establish success or failure. Check the command's exit status and expected
window or image. Adapter enumeration shows available hardware, but does not
prove which adapter presented a particular window.

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

## Local What-if Preview reports a source identity mismatch

Restart the Visualizer from the intended checkout, then enable preview again.
A Visualizer left open while its checkout changes retains the earlier source
identity. The preview worker must use the same checkout, package version and
Git revision. See [Interactive recomputation](../visualizer/interactive_recomputation.md)
for the supported preview workflow.

## Mitsuba and Dr.Jit backend messages

A backend check or Generator run may show messages from Mitsuba and Dr.Jit.
Interpret each message together with the result of the command.

| Message begins with | What it means | When it is acceptable |
|---------------------|---------------|-----------------------|
| `jitc_llvm_init()` | Dr.Jit could not load a usable LLVM API for CPU execution. | The intended backend is CUDA and the command exits successfully. |
| `jit_registry_shutdown()` | Dr.Jit found Mitsuba objects still registered while Python was closing. | The command exits successfully and writes the expected output. |
| `jit_malloc_shutdown()` | Dr.Jit found memory still allocated while Python was closing. | The command exits successfully and writes the expected output. |

Treat the run as failed if it raises an exception, returns a nonzero exit
status or does not produce its expected output. After generation,
[inspect a saved frame](../shared/frame_inspection.md) to confirm that it loads.

Inspect which backend the environment selects when importing Sionna RT:

```bash
python -c "import sionna.rt; import mitsuba as mi; print(mi.variant())"
```

- A variant beginning with `cuda_` means that
  [Sionna RT](https://nvlabs.github.io/sionna/) is using the GPU. In
  that case, an LLVM initialization message concerns the unused CPU backend.
  It does not indicate a CUDA failure when generation succeeds.
- A variant beginning with `llvm_` means that Sionna RT is using the CPU. If
  LLVM cannot initialize, CPU generation cannot proceed. Follow the
  [Linux CPU setup](../getting_started/installation.md#linux-cpu-ray-tracing)
  or [Apple Silicon setup](../getting_started/installation.md#apple-silicon-macos)
  for the library selection and checks.
- If importing Sionna RT fails before a variant is reported, repair the intended
  backend rather than ignoring the initialization message.

After a failed run, investigate the earlier exception before shutdown warnings.
A missing or incompatible LLVM library matters for CPU execution. An otherwise
successful CUDA run does not require the unused CPU backend to initialize.

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

In PowerShell, clear an adapter selection inherited from the shell:

```powershell
$env:PYGFX_WGPU_ADAPTER_NAME = $null
```

Open the scenario using the default adapter choice:

```powershell
orchav-visualizer --renderer pygfx `
  --pygfx-present-method screen `
  --scenario path/to/scenario
```

Close that window, then compare an explicitly selected adapter:

```powershell
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

Omission selects `screen` on Windows and `auto` elsewhere. Begin with that
default on a local desktop or Windows RDP. For Linux VNC, `bitmap` is a useful
compatibility choice. Compare methods with the same scenario and window size.

Force the Linux VNC-compatible path with:

```bash
orchav-visualizer --renderer pygfx \
  --pygfx-present-method bitmap \
  --scenario path/to/scenario
```

Explicit `auto` delegates to rendercanvas, so it differs from omission on
Windows. If `screen` initialization fails, ORCHAV retries once with `bitmap`.
It does not change methods automatically because playback is slow. The option
does not affect Open3D, which can use software rendering in a Linux VNC session.

The status bar and **System → Performance** report server-side frame updates,
render callbacks and renderer submission time. They do not measure remote
desktop encoding, network delivery or client-visible frame rate.

## Performance issues

First distinguish generation time from recorded playback. For generation,
review the ray budget and enabled interactions in
[simulation configuration](../generator/configuration.md). For playback, reduce
the displayed MPC count if drawing is the bottleneck. Use
[cache and playback tuning](../visualizer/caching.md) to choose memory budgets
and diagnose loading costs.

For a controlled renderer workload, use the
[Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md).

## OSM Vegetation: ETH Canopy Tile Fails To Download

- The ETH Global Canopy Height tiles live on
  `https://share.phys.ethz.ch/~pf/nlangdata/ETH_GlobalCanopyHeight_10m_2020_version1/3deg_cogs/`.
  Anonymous HTTPS requires no authentication, but the ETH server occasionally rate-limits
  or goes down for maintenance.
- The Generator logs `ETH canopy tile download failed (...)` and falls
  back to the 8 m default height for every tree. Other vegetation features
  (tree rows, WorldCover gap-fill) continue to work.
- Preserve working canopy tiles and retry the build later. If a particular cached tile is corrupt, remove only that tile and
  rebuild with `OSM_REFRESH_CACHE=1` once the server is back. Tile files
  already cached locally are not re-downloaded.

## OSM Vegetation: WorldCover Reports Water Not In OSM

- If a scene emits
  `WorldCover reports N water cells (~X m²) not present in OSM`, the
  ESA WorldCover raster contains a permanent-water cell that no OSM
  polygon covers.
- Common causes: small ponds, canal segments, or reservoirs that OSM
  volunteers haven't mapped. The Generator does **not** auto-extend
  the water mesh. The warning reports a difference between the source datasets.
- To fix the scene, add the missing water polygon to OSM upstream, or
  turn off the validation with `scene.osm.worldcover_water_check: false` if that comparison is not
  needed. This hides the check and does not add water geometry.

## OSM scene generation is slow or Overpass times out

Overpass can reject requests when a server is busy or unavailable. Keep existing
scenes and working caches, and retry later. Repeated builds or a larger timeout
do not make a busy server accept a query.

Check `osm_metadata.json` and `osm_acquisition_attempt.json` for incomplete
layers and failure details. Required-data failures stop the build. Failed
optional layers can be retried later. Use the
[OSM reuse and recovery guide](../generator/geography/osm_scene_generation.md#reuse-and-recovery)
for timeout settings, cache location and explicit refresh controls. Refresh
bypasses scene reuse, so it is not a remedy for a temporary service outage.

---

Home: [Documentation](../README.md) | Related: [Installation](../getting_started/installation.md) | [Application Configuration](../reference/application_configuration.md)
