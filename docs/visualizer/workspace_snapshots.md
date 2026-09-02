# Workspace Snapshots

A workspace snapshot is a local resume point for the Visualizer. Use one when
you want to continue inspecting the same scenario with the same frame, camera,
filters, and visible layers.

The snapshot references a scenario. It does not contain `scenario.yaml`, scene
assets, or frame data. When a snapshot is opened, the Visualizer loads the
referenced scenario and obtains frames through its configured
[frame provider](../reference/glossary.md#frame-provider). A running Generator
or remote frame server must still be available when that provider requires
one.

In short, `scenario.yaml` defines the scenario and a workspace snapshot records
how that scenario was being viewed.

## What A Snapshot Keeps

| Saved workspace state | Rebuilt, reloaded, or reset |
|-----------------------|-----------------------------|
| Scenario reference and current frame | Scenario YAML, scene assets, and frames are not embedded |
| Camera mode, position, target, up vector, and field of view | Camera matrices and GPU resources are rebuilt |
| Selected TX/RX, node colors, and custom labels | Scene, target, coverage, and MPC geometry are rebuilt from the scenario and frame |
| MPC visibility, coloring, filters, and render limits | A Live Generator or remote frame server is not started |
| Global and per-entry visibility and label choices | Temporary object selections and highlights are reset |
| Trajectory, coverage, aperture, and supported display controls | Runtime material edits, lighting, IBL, shadows, and renderer resources are reset |

[Scenario defaults](scenario_defaults.md) and saved material presets are the
durable way to define appearance. Playback resumes at the saved frame but does
not start automatically.

## Save, Open, And Resume

| Action | How to use it | Behavior |
|--------|---------------|----------|
| Save manually | **File > Workspace > Save Workspace Snapshot...** or `Ctrl+Alt+S` | Writes a named `.json` snapshot. |
| Open manually | **File > Workspace > Open Workspace Snapshot...** or `Ctrl+L` | Validates the snapshot, opens its scenario when needed, then restores supported state. |
| Open a recent snapshot | **File > Workspace > Recent Workspace Snapshots** | Lists the scenario, saved frame, save time, and whether the snapshot is automatic or manual. |
| Save automatically | Close the Visualizer normally with a scenario open | Updates one rolling automatic snapshot for that scenario. Manual snapshots are retained. |
| Resume a named scenario | Launch with `--scenario PATH` | Opens the scenario and restores its newest matching automatic snapshot. A scenario directory and its `scenario.yaml` file identify the same scenario. |
| Resume the last workspace | Launch without `--scenario` | Restores the newest valid automatic snapshot. If none is usable, the Visualizer opens without a scenario. |
| Start cleanly | Add `--no-resume` | Skips automatic restoration for this launch. Manual snapshots can still be opened. |
| Run deterministic output | Use `--benchmark` or `--render-frames` | Disables automatic restoration. |

Snapshots are stored under `~/.orchav/sessions/`. They are local, versioned
files. The Visualizer rejects an unsupported snapshot format or a snapshot
whose referenced scenario no longer exists before changing the current
workspace.

## Practical Rules

- Use a workspace snapshot to resume an investigation on the same machine.
- To share or reproduce a result, share the scenario, its generated frame set,
  and any required assets. A snapshot alone is not sufficient.
- Use `--no-resume` for startup diagnostics, clean captures, and benchmark
  baselines.
- Keep the scenario at the referenced local path, or create a new snapshot
  after moving it.

---

Up: [Visualizer](README.md) | Related: [CLI Reference](cli_reference.md) | [Visualizer Scenario Defaults](scenario_defaults.md) | [Shared Data Layer](../shared/README.md)
