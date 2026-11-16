# ORCHAV Sensing Parameter Reference

Configure ORCHAV Sensing in the `sensing:` block of `scenario.yaml`. Start with
[Acquisition modes](concepts.md#acquisition-modes-and-timing) for acquisition choices or
[Output Control](#output-control) for data needed by later processing.

## Grouped Configuration

Group settings by their role. These authoring groups resolve to the same fully
specified processing recipe used by saved results and replay. Existing flat
YAML names remain accepted. Declaring the same setting twice is an error.

```yaml
sensing:
  enabled: true
  preset: coherent_hann
  acquisition:
    bandwidth_hz: 500.0e6
    frequency_samples: 128
    sample_rate_hz: 1000
    samples_per_cpi: 64
  processing:
    range:
      interpretation: monostatic
  detection:
    cfar:
      threshold_scale: 2.0
    filters:
      minimum_snr_db: 3.0
  recording:
    profile: maps
    detection_stages: true

generator_summary:
  enabled: true
  create: [sensing]
  sensing:
    figures: [range_profile, range_doppler, doppler, truth, detection]
    range_doppler_views: [zoom_to_data, full]
```

`coherent_hann` selects **processing choices**: coherent CPI, CFR/IFFT,
a 512-point range IFFT, Hann frequency/Doppler windows, and weighted-mean
clutter suppression. Bandwidth, measured frequency count, slow-time rate and
CPI sample count belong to `acquisition`. They are not selected by the preset.
Range interpretation is configured separately. Carrier frequency and arrays
remain under `raytracing`.

Individual fields override the preset. Detector sensitivity, tracking, noise
and retention retain their independent defaults. Existing flat configurations
keep the defaults in the tables below.

New grouped `coherent_cpi` configurations default to `windowed_mean` clutter
suppression. Set `processing.clutter.coherent_mode: none` to observe stationary
reflectors or compare unsuppressed signals. The stationary and moving cube examples explicitly select `none`. Weighted mean removal also
attenuates very slow targets. It is a processing choice, not a physical absence
of their returns. `unweighted_mean` names the conditional ordinary-mean
operator. Older `legacy` inputs and saved identifiers still resolve to that
same operator. New YAMLs use the descriptive name.

At 500 MHz this frequency grid repeats every 38.37 m. Its range sampling is
0.075 m, while the bandwidth scale is 0.30 m. Faster targets may require a
higher slow-time rate. These are small-example settings, not universal radar
parameters. Carrier frequency and antenna arrays stay under `raytracing`.

| Group under `sensing` | Configures |
|---|---|
| `acquisition` | `mode`, `bandwidth_hz`, `frequency_samples`, `sample_rate_hz`, `samples_per_cpi` |
| `processing.range` | `interpretation`, `model`, `frequency_window`, `fft_size`, advanced `delay_window` |
| `processing.doppler` | `window`, `fft_size` for modes without a fixed coherent CPI; `hop_samples` for sequential result cadence |
| `processing.clutter` | `coherent_mode`, separate `history_enabled` / `history_window` |
| `processing.path_selection` | `enabled`, `azimuth_min_deg` / `azimuth_max_deg`, `elevation_min_deg` / `elevation_max_deg`. Selects native paths before processing |
| `noise.target_relative` | `enabled`, `snr_db`, `seed`. The bin-floor noise model |
| `noise.frequency` | `variance`, `seed`. Noise on CFR observations |
| `detection` | `enabled`, `mode`, `angle_oversampling` |
| `detection.cfar` | `type`, `edge_policy`, `threshold_scale`, `os_rank`, `range_profile.guard_cells` / `reference_cells`, `range_doppler.guard_range_cells`, `guard_doppler_cells`, `reference_range_cells`, `reference_doppler_cells` |
| `detection.exclusions` | `doppler_edge_bins`, `doppler_dc_bins` before CFAR |
| `detection.filters` | `minimum_snr_db`, `maximum_dynamic_range_db`, `minimum_range_m`, `maximum_range_m` |
| `detection.clustering` | `range_gate_m`, `velocity_gate_m_s` |
| `detection.shortest_range_filter` | `velocity_gate_m_s`, `azimuth_gate_deg`. A scenario-specific duplicate heuristic, not general multipath rejection |
| `tracking` | `enabled`, `type`, `confirmation.hits` / `window`, `max_missed_updates`, `output_tentative`, `acceleration_psd_m2_s3`, `association.range_gate_m`, `velocity_gate_m_s`, `azimuth_gate_deg`, `position_gate_m` |
| `evaluation` | `range_gate_m`, `velocity_gate_m_s` for truth scoring only |
| `recording` | `profile`, `display`, `detection_stages`, `tracking_diagnostics`. Established explicit controls: `range_doppler`, `range_profile`, `complex_observations` |

For coherent acquisition, `sample_rate_hz` and `samples_per_cpi` also supply the
corresponding ray-tracer CIR settings. Conflicting explicit CIR settings are
rejected. The scene `timeline` remains a separate clock. Grouped CFR/IFFT
selects the required delay-domain `boxcar` taper when it is omitted. The
frequency window is configured separately.

For sequential acquisition, set `acquisition.mode: sequential_stft`, use
`processing.doppler.fft_size` for the analysis-window length, and keep
`timeline.sample_rate_hz` equal to `acquisition.sample_rate_hz`.
Set `processing.doppler.hop_samples` to the number of pulses between result
windows. It defaults to 1 and must be an integer from 1 through the Doppler
window length. For example, a 256-sample window with `hop_samples: 64` has
75% overlap. Every pulse still updates the acquired history; the hop controls
when a complete range–Doppler result is emitted. This option applies to
`sequential_stft`; coherent and analytic acquisition retain their own cadence.
The ray tracer supplies one CIR sample per solve. Omit
`raytracing.cir_sampling_frequency_hz`. Sequential CFR/IFFT supports SISO
and uses the same grouped range and frequency-noise settings as coherent
acquisition. Explicit `processing.clutter.coherent_mode` applies to each
complete CFR window, `processing.clutter.history_enabled: true` is rejected.

Sequential acquisition requires uniformly timed pulses at the configured rate.
A missing pulse invalidates later windows. Resolve the failure and restart the
run to obtain a fresh uniform sequence. See [result timing](concepts.md#result-timing)
for warm-up, overlapping windows and scene binding.

Figure options belong to `generator_summary.sensing`, including optional
`inspection_figures: [detection_stages, tracking_decisions]`. The default figure groups are `range_profile`, `range_doppler`, `doppler`,
`truth`, `detection` and `angles` when available. Add `acquisition` to the
`figures` list to generate the acquisition/processing diagram.
`doppler` produces one time–Doppler view from saved CPI spectra, or a spectrum
when there is only one CPI. It adds no clutter suppression or signal transform.

Native figures are written to `summary/sensing/` beside the scenario YAML.
The following display settings also belong to `generator_summary.sensing`:

| Setting | Default | Description |
|---|---|---|
| `history_time_axis` | `scene_frame` | `time` uses saved result timestamps in seconds for range/velocity truth and tracking histories, `scene_frame` uses scene-frame indices. Doppler and Cartesian histories use physical time. |
| `range_profile_views` | `[zoom_to_data, full]` | Select either or both range-axis views of the last-sample range profile. |
| `range_doppler_views` | `[zoom_to_data, full]` | Select either or both focused and full range–Doppler views. |

These options change the figures only. See
[generated sensing figures](../generator/generated_figures.md#sensing-figures)
for native-summary YAML and optional regeneration from saved results.

Omit view bounds to use automatic zoom-to-data. Summary exports also include a
full-grid view. Optional fixed bounds belong to `view_defaults.sensing.range_xlim`
and `velocity_ylim`. They do not filter detections or crop recorded arrays.
The tables below use the flat names stored in resolved recipes. For the main
sampling controls, the correspondence is:

| Authoring field under `sensing` | Resolved field |
| --- | --- |
| `acquisition.mode` | `processing_mode` |
| `acquisition.sample_rate_hz` | `prf_hz` |
| `acquisition.bandwidth_hz` | `bandwidth` |
| `processing.range.fft_size` | `fft_size_range` |
| `processing.doppler.fft_size` | `fft_size_doppler` |
| `processing.doppler.hop_samples` | `cpi_hop_samples` |

For coherent acquisition, `acquisition.samples_per_cpi` sets the Doppler length
and ray-tracer CIR sample count together. The Generator saves the complete
effective `config_resolved`. Consumers use that saved record rather than filling
missing scientific settings from these tables. See the
[sensing frame extension](../shared/frame_reference.md#sensing-payloads)
for storage and compatibility.

## Delay Grid And Range

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enables sensing product generation. |
| `bandwidth` | `100e6` | Bandwidth in Hz. For Delay-bin accumulation (`point_bin_floor`), sets nominal monostatic grid spacing `c/(2B)`. It does not instantiate a waveform or matched filter and is separate from `raytracing.bandwidth_hz`. |
| `prf_hz` | `2000.0` | Pulse repetition frequency in Hz. |
| `range_mode` | `bistatic` | `bistatic` uses total path length. `monostatic` reports round-trip distance divided by two. |

Carrier frequency is configured once at `raytracing.carrier_frequency_hz`.
Native sensing powers are relative and unitless. These parameters do not define
calibrated received power, a link budget, thermal receiver noise, or radar cross
section.

## FFT And Windows

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_size_range` | `256` | Delay-bin accumulation: retained CIR bin count. Increasing it extends the span without refining spacing or adding bandwidth. CFR/IFFT: IFFT length M, with M >= `range_response.num_frequency_bins`. |
| `fft_size_doppler` | `128` | Doppler FFT size. |
| `cpi_hop_samples` | unset | Sequential acquisition: pulses between consecutive result windows, from 1 to `fft_size_doppler`. Omission uses one pulse. Set through `processing.doppler.hop_samples`; not applicable to coherent acquisition. |
| `window_range` | `hann` | Delay-bin accumulation: pointwise taper over CIR bins, without a range FFT. CFR/IFFT requires `boxcar`. Its spectral taper is `range_response.frequency_window`. |
| `window_doppler` | `hann` | Slow-time Doppler window name. |

See [Choose a range-response model](concepts.md#range-response-models)
for the conceptual difference between delay-bin accumulation and CFR/IFFT.

## Processing Mode

| Parameter | Default | Description |
|---|---|---|
| `processing_mode` | `auto` | Choose `coherent_cpi` or `sequential_stft`. The default resolves to coherent acquisition when `raytracing.cir_time_steps > 1`. Specify the mode explicitly for a single CIR time step. |

See [acquisition modes](concepts.md#acquisition-modes-and-timing) for the array, sampling and timing requirements.

## Optional Finite-Bandwidth Response

`range_response` selects Delay-bin accumulation (`point_bin_floor`, default)
or Frequency-domain channel synthesis (CFR/IFFT) (`cfr_ifft`). Generation and
CPU replay share this configuration. The
[range-response guide](concepts.md#range-response-models) compares the models.
The [windowing exercise](../../scenarios/sensing/foundations/windowing/README.md)
provides runnable commands for changing frequency weighting on saved observations.

| Key | Default | Meaning |
|---|---|---|
| `range_response` | omitted | Typed response configuration. Omission selects Delay-bin accumulation. |
| `range_response.model` | `point_bin_floor` | Select `point_bin_floor` or `cfr_ifft`. CFR/IFFT supports coherent acquisition and SISO `sequential_stft`. |
| `range_response.num_frequency_bins` | unset | Required N >= 2 for CFR. Frequency spacing is `bandwidth / N`. |
| `range_response.frequency_window` | `boxcar` | `boxcar` or periodic `hann`, with unit coherent-peak normalization. |
| `range_response.frequency_noise.variance` | `0.0` | Nonnegative circular complex variance per frequency observation. |
| `range_response.frequency_noise.seed` | unset | Explicit nonnegative integer required for nonzero variance. |

For CFR/IFFT, `fft_size_range` is IFFT length M >= N and `window_range` must
be `boxcar`. Delay sample spacing is N/(M B). The bandwidth resolution scale
remains 1/B. Exact range/path-length axes are saved. Use `frequency_noise`
instead of the incompatible `noise_enabled`/`noise_snr_db`/`rng_seed` controls.
The `reprocessing` and `full` profiles retain the actual unwindowed noisy observation.
The explicit `raw_payload_policy` settings `diagnostic` and `always` do so too.
Without that observation, frequency-window/IFFT replay is unavailable. Omitting the model or explicitly
selecting `point_bin_floor` gives the same acquisition identity.

Sequential CFR noise is sampled once per acquired pulse and reused in every
overlapping window containing it. A full Doppler window is required even
though a single pulse can already produce a range profile. Retained warm-up
observations remain incomplete acquisitions on replay. The ready result uses
the midpoint of its sample timestamps, and its range profile uses the last
sample before slow-time clutter removal.

## CFAR Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfar_type` | `ca` | CFAR detector type. `ca` is cell averaging. |
| `cfar_edge_policy` | `full_window` | `full_window` requires every configured training cell, `truncate` explicitly uses available edge cells. |
| `cfar_guard_cells` | `4` | 1D guard cells for profile-style detection. |
| `cfar_ref_cells` | `16` | 1D reference cells for profile-style detection. |
| `cfar_threshold_scale` | `3.0` | Raw linear-power multiplier on the CA mean or OS order statistic. It is not dB or a requested probability of false alarm. |
| `cfar_os_rank` | `0.75` | Ordered-statistics rank fraction when OS CFAR is selected. |
| `cfar_rd_guard_cells_range` | `2` | Range guard cells for 2D range-Doppler CFAR. |
| `cfar_rd_guard_cells_doppler` | `1` | Doppler guard cells for 2D range-Doppler CFAR. |
| `cfar_rd_ref_cells_range` | `8` | Range reference cells for 2D range-Doppler CFAR. |
| `cfar_rd_ref_cells_doppler` | `4` | Doppler reference cells for 2D range-Doppler CFAR. |
| `cfar_min_snr_db` | `6.0` | Minimum local signal-to-background contrast in dB for a valid detection. This is not calibrated receiver SNR. |
| `cfar_max_dynamic_range_db` | `None` | Optional power span below the strongest cell per detector input/beam. Coherent beam scans also use the recorded pre-clutter peak as a lower bound on this reference. Effective threshold is `max(CFAR threshold, reference_peak × 10^(-span/10))`. A sensitivity limit, not calibrated noise. Weak real echoes can be rejected. |
| `pre_cfar_doppler_edge_guard_bins` | `2` | Doppler edge bins suppressed before CFAR. |
| `pre_cfar_doppler_dc_guard_bins` | `0` | Total width of the zero-Doppler exclusion, in bins. Not a radius on each side. |
| `min_detection_range_m` | `null` | Optional lower range gate. |
| `max_detection_range_m` | `null` | Optional upper range gate. |

For an FFT-shifted Doppler axis, a DC guard of `3` suppresses the zero bin
and one bin on either side. A width of `2` suppresses zero and the adjacent
negative bin. The positive adjacent bin remains eligible. Prefer an odd width
when a symmetric exclusion is intended. This changes detector input only.
The stored RD product remains intact. A wider exclusion also prevents detection
of genuine slow or stationary returns within that band. Edge guards count bins
at **each** Doppler extreme.

See [how detector stages select and filter candidates](detection_and_tracking.md#what-each-stage-decides).

## Tracking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_confirmation_enabled` | `false` | Enables multi-frame confirmation. |
| `track_confirmation_m` | `2` | Recent hits required for initial confirmation in all built-in trackers. |
| `track_confirmation_n` | `3` | Recent-update window for initial confirmation in all built-in trackers. |
| `track_confirmation_mode` | `latched` | `latched` keeps confirmation until retirement, `continuous` rechecks M-of-N every update and can deconfirm after misses. Grouped YAML: `tracking.confirmation.mode`. |
| `track_max_missed_frames` | `1` | Misses tolerated before a track is dropped. |
| `track_association_range_gate_m` | `0.6` | Range gate for associating detections to tracks. |
| `track_association_velocity_gate_m_s` | `0.3` | Velocity gate for associating detections to tracks. |
| `track_association_position_gate_m` | `2.0` | Positive receiver-local Euclidean position gate in metres for associating reports with Cartesian track predictions. Applies in x/y or x/y/z according to tracker type. |
| `track_acceleration_psd_m2_s3` | `1.0` | Nonnegative per-axis acceleration-noise power spectral density in m²/s³ for Cartesian Kalman prediction. |
| `track_output_tentative` | `false` | Emit tentative tracks before confirmation. |
| `tracker_type` | `m_of_n` | `m_of_n`, `kalman`, `jpdaf`, `cartesian` (x/y), or `cartesian3d` (x/y/z). |

For tracker behavior and limitations, see
[detection and tracking](detection_and_tracking.md).

## Evaluation Gates

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_association_range_gate_m` | `10.0` | Range tolerance for TP/FP/FN association metrics. |
| `eval_association_velocity_gate_m_s` | `5.0` | Velocity tolerance for TP/FP/FN association metrics. |

These gates associate detector outputs with ground truth only when computing
evaluation metrics. They do not change CFAR detections and are independent of
the tighter `track_association_*` gates used to update tracks.

## Clutter And Noise

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clutter_removal_enabled` | `false` | Enable sliding-baseline history removal in point-bin `sequential_stft`. Sequential CFR/IFFT rejects `true`. |
| `clutter_removal_window` | `8` | Sequential history window length. |
| `coherent_clutter_mode` | `legacy` | Coherent CPI and sequential CFR clutter operator: `none`, `windowed_mean` or `unweighted_mean` (alias of `legacy`). The compatibility behavior is explained below. Raw products are unaffected. |
| `noise_enabled` | `false` | Enables target-relative complex AWGN in the relative unitless sensing sample domain. This is not thermal or link-budget noise. |
| `noise_snr_db` | `null` | SNR relative to mean nonzero-input-sample power. Required when `noise_enabled` is true. |
| `rng_seed` | `null` | Optional random seed for deterministic noise products. |

For backward compatibility, an omitted coherent mode retains the historical primary policy: it subtracts each range bin's
slow-time mean only when sampled rows have nonzero variation, leaving identical
rows unchanged. Coherent recipes can explicitly select `windowed_mean` to
remove constant clutter too and account for the Doppler taper. Its recorded
policy is `window_weighted_mean`. `clutter_removal_enabled` is not the switch
for either CPI projection. Likewise, postprocessor `--no-mti` affects only rebuilt or
postprocessed products. It does not change the generator-authored primary map.

The scenario validator warns when a coherent-CPI recipe omits this choice.
The warning does not change the effective mode or reject the scenario.

See [clutter processing](concepts.md) for the effect on near-zero Doppler.

## Angle Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aoa_filter_enabled` | `false` | Enables path filtering by angle of arrival before CIR construction. |
| `aoa_az_min_deg` | `null` | Minimum wrapped azimuth in degrees. |
| `aoa_az_max_deg` | `null` | Maximum wrapped azimuth in degrees. |
| `aoa_el_min_deg` | `null` | Minimum elevation in degrees. |
| `aoa_el_max_deg` | `null` | Maximum elevation in degrees. |

## Output Control

Use `sensing.recording.profile` to choose retained data for a new scenario.
[Recording profiles](reprocessing.md#choose-a-profile) explains the four choices
and which later processing each supports. A profile cannot be combined with
`complex_observations` or `raw_payload_policy`. The explicit controls below
remain available when no profile is selected.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recording_profile` | `null` | Flat form of `recording.profile`: `results`, `maps`, `reprocessing` or `full`. Omitted profiles retain the explicit output controls below. |
| `recording_display` | `null` | Flat form of `recording.display`: fixed motion-display recipe for profiles that retain numeric plots. See [recording profiles](reprocessing.md#motion-display-products). |
| `output_range_doppler` | `true` | Store range-Doppler products. |
| `output_range_profile` | `false` | Store range profile products. |
| `output_detections` | `true` | Store detection outputs. |
| `persist_detection_stage_artifacts` | `false` | Persist typed CFAR input/noise/threshold/count/eligibility/mask, NMS, SNR/value-validity, and cluster-label products with lineage. |
| `persist_tracking_diagnostics` | `false` | Save actual tracking predictions, candidate costs, assignments and lifecycle for optional inspection. |
| `raw_payload_policy` | `none` | Explicit array storage for configurations without a profile: `none` keeps only processed products, `diagnostic` also keeps available CIR, slow-time and frequency observations, and `always` keeps every available raw array. |

Detector-stage retention saves the complete ten-stage graph with recipe and
source lineage. Partial stage sets are rejected. The postprocessor's
`--save-stage-artifacts` writes these products into derived frames plus a
compressed NPZ and applicable JSON sidecar. See [inspection options](#optional-inspection-evidence)
for the corresponding figure controls.

## Figure Display Hints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `display_range_xlim` | `null` | Optional `[min_m, max_m]` range axis for summary figures. |
| `display_velocity_ylim` | `null` | Optional `[min_m_s, max_m_s]` velocity axis for summary figures. |

## Complete configurations

Use the [stationary cube](../../scenarios/sensing/cube_radar/stationary/scenario.yaml)
for coherent acquisition and the [oscillating cube](../../scenarios/sensing/cube_radar/oscillating_sequential/scenario.yaml)
for sequential acquisition. Their walkthroughs explain the physical settings
and expected outputs. [Acquisition modes](concepts.md#acquisition-modes-and-timing)
explains the timing relationship without duplicating those scenario files.

## Beamwise Detection And Angular Association

| Parameter | Default | Description |
|---|---|---|
| `array_detection_mode` | `rd` | `beam_scan` enables independent per-beam detection before combining candidates. Requires a monostatic analytical-Doppler array acquisition. |
| `beam_angle_oversampling` | `1` | Integer from 1 to 16. With `beam_scan`, factors above 1 refine angles by locally zero-padding the complex spatial FFT before tracking. No additional RT solves. This reduces angle-grid quantization without increasing aperture resolution. |
| `cluster_range_gate_m` | `1.0` | Range separation used by candidate clustering, in m. |
| `cluster_velocity_gate_m_s` | `5.0` | Radial-velocity separation used by candidate clustering, in m/s. |
| `ghost_filter_velocity_gate_m_s` | disabled | Velocity tolerance for nearer-return ghost suppression, in m/s. |
| `ghost_filter_azimuth_gate_deg` | disabled | Also require azimuth agreement. Needs velocity ghost filtering and angular candidates. |
| `track_association_azimuth_gate_deg` | disabled | Maximum azimuth separation used by angular tracker association, in degrees. |

See [beamwise detection](detection_and_tracking.md#beamwise-detection) for supported inputs,
field of view, saved-stage interpretation, and limitations. Clustering,
tracker association, ghost suppression, and evaluation gates serve different
purposes. Evaluation gates never feed detections back into the tracker.

## Optional Inspection Evidence

| Native YAML | Postprocessor | Default | Purpose |
|---|---|---|---|
| `sensing.persist_detection_stage_artifacts` | `--save-stage-artifacts` | false | Retain tested power, threshold and detector masks/labels |
| `sensing.persist_tracking_diagnostics` | `--save-tracking-diagnostics` | false | Retain prior/posterior, candidate costs, assignments and lifecycle |
| `generator_summary.sensing.inspection_figures` | `--figure detection_stages` / `--figure tracking_decisions` | empty | Render selected inspection figures |

Native figures also require `generator_summary.create: [sensing]`. Retention
and rendering are independent: enabling records does not require plotting.
Candidate records scale with track/report pairs. Enable them for diagnosis.
Older results display missing details as “not recorded.” See
[detection](detection_and_tracking.md#detection) and [tracking](detection_and_tracking.md#tracking).

### Exact JPDA controls

The corresponding flat schema/runtime names are `jpda_max_joint_nodes`,
`jpda_birth_confirmation_frames`, and `jpda_birth_separation_chi2`. Runtime constructor options use the same names.

| YAML under `sensing` | Flat runtime/schema field | Postprocessor CLI | Default |
|---|---|---|---:|
| `tracking.jpda.max_joint_nodes` | `jpda_max_joint_nodes` | `--jpda-max-joint-nodes` | 100000 |
| `tracking.jpda.birth_confirmation_frames` | `jpda_birth_confirmation_frames` | `--jpda-birth-confirmation-frames` | 3 |
| `tracking.jpda.birth_separation_chi2` | `jpda_birth_separation_chi2` | `--jpda-birth-separation-chi2` | 1 |

The search limit must be a positive integer. Birth persistence must be at least
two updates. An exceeded joint-search budget rejects the update atomically.
Initial confirmation uses a recent M-of-N window in every built-in tracker.
Confirmation then follows the selected lifecycle policy.

Birth separation is an R-normalized duplicate-suppression policy, not physical
resolution. It must be finite and nonnegative. Zero suppresses exact duplicates only.

### Track confirmation lifecycle

M-of-N, Kalman, Cartesian 2-D/3-D, and JPDAF all use recent M-of-N hits
for initial confirmation. By default, confirmation is latched for the lifetime
of that track ID. Misses still age the track: it is deleted when consecutive
misses exceed `track_max_missed_frames`. A new track must confirm again.
Tentative tracks cannot accumulate arbitrarily old hits to confirm. JPDAF counts
a hit when the marginal missed-detection probability is at most 0.5.

Confirmed coasting states remain internal (or explicitly labelled in optional
tracking diagnostics). They do not create predicted-only detection rows.
`track_output_tentative` and the minimum cumulative-hit output gate remain
independent of confirmation.

For postprocessing, use `--tracker-confirmation-mode latched|continuous` or
`PostProcessingConfig(tracker_confirmation_mode="continuous")`. Direct tracker
users select `TrackerConfig(confirmation_mode="continuous")`. The default is
`latched` at every entry point. The selected policy is recorded in tracker
metadata and resolved configuration for reproducible comparisons.
