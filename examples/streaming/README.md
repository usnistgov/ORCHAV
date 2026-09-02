# Streaming Examples

[`streaming_examples.py`](streaming_examples.py) contains small Python classes
for custom Live Generator (gRPC) workflows:

- `LiveSensorMobility`: poll a live sensor-like source with a local cache.
- `APIMobility`: fetch positions from an HTTP endpoint and fall back to a
  deterministic simulated path when the endpoint is unavailable.
- `PhysicsSimulationMobility`: integrate a simple position/velocity model.
- `RandomWalkMobility`: generate a bounded random walk.
- `StreamingLookAtOrientation`: orient a device toward a streaming target.
- `StreamingCircularOrientation`: rotate yaw, pitch, or roll over time.

These are not standalone scenario files. Copy the class you need into a
Python-scripted scenario or import it from a source checkout, then pass it into
your Generator configuration where a `MobilityPattern` or prepared orientation
source is expected. The example-local `StreamingOrientationSource` implements
the Generator's small `prepare()`/`orientations()` protocol and delegates spin
and look-at evaluation to the canonical quaternion kernel. It is not part of
the core Generator API. For mobility sources that
naturally produce one position per frame, `streaming_examples.py` includes a
local `StepwiseMobility` helper that adapts `get_position(step)` to the
Generator's batch `get_positions(...)` contract.

## Minimal Smoke Test

```bash
python -c "from examples.streaming.streaming_examples import RandomWalkMobility; mobility = RandomWalkMobility(start_position=(0.0, 0.0, 1.5), step_size=0.2); print(mobility.get_position(0)); print(mobility.get_position(1))"
```

This form works from Bash, PowerShell, and Command Prompt when run from the
repository root.

## Where It Fits

Custom streaming patterns are useful when the device or target state is not
known from a static YAML timeline. Typical examples are live sensors, external
simulators, or interactive Live Generator experiments.

For the supported end-to-end data modes, start with:

- [Choose a Data Mode](../../docs/shared/README.md#choose-a-data-mode)
- [Data Mode Scenarios](../../scenarios/visualizer/data_modes/README.md)
- [Scenario Authoring](../../docs/generator/scenario_authoring.md#python-scripted-actors)

---

Up: [Examples](../README.md)
