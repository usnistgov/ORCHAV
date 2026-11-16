# Custom Tracker Extensions

Replace tracking through `SensingPostProcessor` while reusing detections,
physical timestamps, result storage, and evaluation. The interface below accepts
saved detections from the maintained CFR/IFFT examples.

## Boundary

`SensingPostProcessor(config, tracker_factory=...)` constructs one tracker per
processor instance. The factory receives the processing configuration. Its
`filter_detections(detections, *, update_time_s)` method receives:

- Candidate dictionaries after detection and configured pruning, including
  `range_m`, `velocity_m_s`, optional `aoa_az_deg`/`aoa_el_deg` and cell indices.
- A finite, strictly increasing result timestamp in seconds on the scenario
  clock, including updates with an empty candidate list.

The monostatic tutorials use monostatic range, positive-closing velocity, and receiver-local
array angles. Cartesian tracking requires an explicit sensor-pose transform.
Bistatic path range/rate requires its own measurement model. The hook receives
no ground truth.

Return `(output_detections, metadata)`. Preserve measurement fields and append
track IDs and estimated state fields using the existing tracker convention.
Do not turn a prediction during an empty update into a measured detection.
Record confirmation and deletion behavior in metadata.

## Minimal Pattern

```python
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor

class ResearchTracker:
    def __init__(self, config):
        self.config = config

    def filter_detections(self, detections, *, update_time_s):
        # Insert timestamp-aware association/state estimation here.
        return detections, {"enabled": True, "tracker_type": "research_passthrough"}

config = PostProcessingConfig()
processor = SensingPostProcessor(config, tracker_factory=ResearchTracker)
```

This pass-through sketch shows the interface. It does not estimate a track.

Use a `PostProcessingConfig` dataclass subclass for algorithm identity/version
and extra parameters. Those fields participate in the processing recipe hash.
Update the recorded version when algorithm behavior changes.
Run custom processors through
`shared.sensing.experiment.compare_recipes(..., processor_factory=your_factory)`
to use the standard typed writer, source acquisition identity and GT evaluation.

Comparisons use each recording's declared [evaluation reference](evaluation.md).
Missing support for tracking metrics is reported by `tracking_unavailable_reason`,
while saved tracks and supported detection metrics remain available.

The hook covers tracking. A custom detector needs a separate integration that
preserves product coordinates, stage provenance, and detection links.

---

Up: [Sensing](README.md) | Related: [Tracking](detection_and_tracking.md#tracking),
[Postprocessing](reprocessing.md#run-processing)
