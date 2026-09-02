# Generator Test Suite

This directory contains tests for the `generator` package.

## Structure

- `unit/`: Fast tests for `core`, `io`, and `viz`. Expensive Mitsuba and
  [Sionna RT](https://nvlabs.github.io/sionna/) dependencies are mocked where
  appropriate.
- `integration/`: Orchestration tests for shared scenario loading and generator
  engine loops, with rendering and output replaced when the test does not need
  them.
- `scenarios/`: Small scenario fixtures used by integration tests.

## Guidelines

- Use pytest. Mark slow tests and keep unit tests isolated from the file system
  and network.
- Prefer deterministic fixtures and mocking for external systems.
- Add new generator tests directly under `tests/generator/`.
- Keep assertions on shapes, counts, and invariants. Avoid depending on real
  rendering.
