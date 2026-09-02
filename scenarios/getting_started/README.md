# Getting Started Scenarios

[Scenarios](../README.md) > Getting Started

These two examples introduce ORCHAV's normal generation, inspection, and
visualization workflow. Both use `etoile`, a
[Sionna RT](https://nvlabs.github.io/sionna/)-provided scene
representing the Arc de Triomphe roundabout in Paris, and the same
one-transmitter/one-receiver idea.

| Scenario | Authoring path | What it teaches |
|---|---|---|
| [`hello_world/`](hello_world/README.md) | YAML only | Validate a minimal scenario, generate one HDF5 frame, inspect it, and open it in the Visualizer. |
| [`hello_world_scripted/`](hello_world_scripted/README.md) | YAML plus `generate.py` | Calculate a curved RX trajectory in Python while keeping shared scene, timeline, and ray-tracing settings in YAML. |

YAML-only authoring is the default. Use a Python driver when a scenario needs
calculated actor specifications, external preprocessing, generated case lists,
or another behavior outside the YAML schema.

## Begin

Start with [Hello World](hello_world/README.md). Its Running section gives the
complete validation, generation, inspection, and visualization sequence. After
that succeeds, the page continues to the first Visualizer session and offers
the scripted companion as an optional authoring branch.

For an installation-first walkthrough, use the
[Quickstart](../../docs/getting_started/quickstart.md).

---

Up: [All Scenarios](../README.md) | Begin: [Hello World](hello_world/README.md)
