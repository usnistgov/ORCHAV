# Custom Motion Sources

[`streaming_examples.py`](streaming_examples.py) contains Python building blocks
for obtaining positions and orientations from a sensor-like source, an API or
a simulation. They are API examples, not complete live-sensor integrations.

- `LiveSensorMobility`: poll a live sensor-like source with a local cache.
- `APIMobility`: fetch positions from an HTTP endpoint and fall back to a
  deterministic simulated path when the endpoint is unavailable.
- `PhysicsSimulationMobility`: integrate a simple position/velocity model.
- `RandomWalkMobility`: add a bounded directional step followed by Gaussian
  noise. The noise and accumulated position are unbounded.
- `StreamingLookAtOrientation`: orient a device toward a streaming target.
- `StreamingCircularOrientation`: rotate yaw, pitch, or roll over time.

The example below connects one position source to a normal scripted scenario.
For a complete script organized into an actor factory and a generation command,
see [Hello World Scripted](../../scenarios/getting_started/hello_world_scripted/README.md)
and its [generate.py](../../scenarios/getting_started/hello_world_scripted/generate.py).

## Generate a Sampled Trajectory

With the ORCHAV environment active, run this Python example from the repository
root. It copies the small scripted scenario's settings into a fresh directory
and samples one receiver position for each of its 16 timeline steps:

```python
from pathlib import Path
from shutil import copyfile

from examples.streaming.streaming_examples import RandomWalkMobility
from generator import build_simulation_config, perform_pipeline
from shared.scenarios import load_scenario_configuration
from shared.scenarios.actors import (
    ActorsSpec, RxActorSpec, SampledMobilitySpec, StationaryMobilitySpec, TxActorSpec,
)

output = Path("analysis/custom-motion")
output.mkdir(parents=True, exist_ok=False)
copyfile(
    "scenarios/getting_started/hello_world_scripted/scenario.yaml",
    output / "scenario.yaml",
)
scenario = load_scenario_configuration(output, project_root=Path.cwd())
source = RandomWalkMobility(start_position=(-27.18, -12.61, 1.5), step_size=0.2)
positions = tuple(source.get_position(i) for i in range(scenario.timeline.steps))
actors = ActorsSpec(
    tx=(TxActorSpec(
        name="TX1",
        mobility=StationaryMobilitySpec(position_m=(-114.0, 37.0, 30.0)),
    ),),
    rx=(RxActorSpec(
        name="RX1", mobility=SampledMobilitySpec(positions_m=positions),
    ),),
)
perform_pipeline(
    simulation_config=build_simulation_config(scenario),
    scenario_configuration=scenario,
    actors=actors,
)
```

The Generator writes HDF5 frames under `analysis/custom-motion/frames/`.
Open them from another terminal:

```bash
orchav-visualizer --scenario analysis/custom-motion --renderer pygfx --no-resume
```

In **Scene → Nodes**, enable the RX **Trajectory** to see the sampled walk.
Each run chooses a new random path; use a fresh output directory for another run.
The walk is illustrative motion, without a collision or physical-velocity model.

This bridge samples the complete trajectory before ray tracing. A live device
that changes while the simulation is running needs an additional runtime
integration. [Live Generator](../../scenarios/visualizer/data_modes/live_grpc/README.md)
describes on-demand frame delivery and its separate server/client commands.

## Runtime Interfaces

For custom integrations, the local `StepwiseMobility` helper adapts
`get_position(step)` to the Generator's batch `get_positions(...)` contract.
`StreamingOrientationSource` implements `prepare()`/`orientations()` and uses
the canonical orientation calculations for spin and look-at behavior. These
example helpers are separate from the immutable actor specifications used by
scenario authoring.

See [Python-Scripted Actors](../../docs/generator/scenario_authoring.md#python-scripted-actors)
to share a calculated trajectory between generation and local preview.

---

Up: [Examples](../README.md)
