# ORCHAV Visualizer Demos

The videos below show ORCHAV’s visualization, channel analysis and sensing
features across several simulated scenes. They use saved recordings to show
the interface and its results, with links to the corresponding controls.

If a GitHub player is unavailable, follow the MP4 link below that video and
choose **Download raw file** to play it locally.
With a downloaded checkout, open `docs/assets/visualizer/feature_films/index.html`
in your browser to play all four bundled videos.

<a id="one-scene-several-ways-to-read-it"></a>

## Eiffel Tower scene and propagation paths

The first video shows an Eiffel Tower simulation with animated multipath
components (MPCs). A camera moves around the scene as it switches between
realistic, radio-material and glow views, then holds still during playback of
the recorded motion. Pulses along the paths are slowed for visual inspection.

https://github.com/user-attachments/assets/12392243-90c8-4697-be55-5a6e6acc1bbf

[Backup MP4: Eiffel Tower demo](../assets/visualizer/orchav_eiffel_mpc_demo.mp4) ·
[Demo mode controls](presentation_tools.md#demo-mode)

<a id="orchav-explore-the-channel"></a>

## Main ORCHAV features

This video shows scene appearance, channel metrics and coverage inspection
across several examples. It includes changes to building facades and material
rendering, channel statistics as actors move, and coverage at different heights.

https://github.com/user-attachments/assets/8758b1ad-4659-47ca-8d4b-c8141bee6317

| Time | What the video shows |
| --- | --- |
| 00:02 | The original Paris facades and two procedural styles applied to the same recorded scene. |
| 00:12 | The effects of roughness, metal and clearcoat under the same lighting. |
| 00:20 | Changes in path count, delay and power as the recorded actors move. |
| 00:30 | Path count and delay spread across a Paris Opéra recording with 24 links. The graphs use all recorded paths, while the scene shows up to 4,000 of the strongest. |
| 00:38 | Coverage at different heights and path-gain thresholds, showing the direct-path radio shadow behind the Burj Khalifa. |

[Backup MP4: main ORCHAV features](../assets/visualizer/feature_films/orchav_features.mp4) ·
[Appearance controls](appearance.md) · [Metrics and coverage](analysis.md)

<a id="orchav-sensing-inspect-the-estimates"></a>

## ORCHAV Sensing features

This video shows sensing results from Paris Opéra, a drone near the Burj
Khalifa and a set of spatial detection trials. It illustrates target tracking,
estimated position uncertainty, range and angle evidence, and comparisons of
matched and missed detections.

https://github.com/user-attachments/assets/9a0bcbdc-75e8-447c-b28a-5ccbec93747d

| Time | What the video shows |
| --- | --- |
| 00:02 | The car and scooter moving through Paris Opéra alongside their saved tracks. |
| 00:12 | The tracker’s estimated position uncertainty around a drone near the Burj Khalifa. |
| 00:20 | Range and angular support for a car’s estimated position in a single radar update. |
| 00:32 | Matched (cyan) and missed (orange) trials by height and in 3D. Each cell shows the outcome of an independent simulated trial. |

[Backup MP4: ORCHAV Sensing features](../assets/visualizer/feature_films/orchav_sensing.mp4) ·
[Sensing Dashboard](sensing_dashboard.md) · [Sensing volumes](volume_inspection.md#sensing-volumes)

## Human target response: RCS and articulated mesh

The fourth video compares a rigid native human RCS model on the left with an
articulated human mesh on the right. The same radar and approaching trajectory
produce a narrow Doppler band for the rigid point model and a broader, changing
signature for the mesh.

At 00:14, the comparison replays with all saved MPCs visible, showing the single
RCS return beside the mesh’s surface paths.

https://github.com/user-attachments/assets/126bc63a-b932-42aa-a20f-f592f323522f

The panels share time and velocity axes. Power is relative to each recording’s
fixed peak, so compare spectral shape rather than absolute received power.
This comparison uses two specific target configurations: ORCHAV's native
Sionna RT 2.2 human RCS model with rigid translation, and an animated
mesh/material human target. It does not establish that all meshes produce
micro-Doppler or that all RCS models necessarily behave as rigid objects.
For the scattering-model distinction and current ORCHAV support, see
[target representations](../sensing/target_models.md#choose-a-representation).

[Backup MP4: human comparison](../assets/visualizer/feature_films/orchav_human_response.mp4)

<details>
<summary>Comparison setup and credits</summary>

Both recordings use 28 GHz, 500 MHz bandwidth, 4,000 pulses/s, 64 ms Hann
windows and a 1.4 m/s approach. The mesh uses SMPL/AMASS CMU `08/08_03`
marching motion and an uncalibrated metal material. The native model is
Sionna’s TR 38.901 human model 2 with one translating scattering point.
Surface visibility and path sampling also affect the mesh spectrum. The saved
acquisitions used Sionna RT 2.0.1 for the mesh and 2.2.0 for native RCS.
The comparison illustrates their target responses without isolating the effects
of the solver version.

</details>

<a id="try-it-yourself"></a>

## Related examples and controls

Generate a recording with the [Quickstart](../getting_started/quickstart.md),
then open it and inspect its paths in [Your First Visualizer Session](first_session.md).

Use [Demo mode](presentation_tools.md#demo-mode) to present your own recordings with camera tours, animated paths and appearance cycling.

## Sources and credits

The videos use saved Eiffel Tower, Paris Opéra and Burj Khalifa simulations,
with ORCHAV procedural facades and roads. The acquisition recordings are not
bundled.
The [Paris showcase](../../scenarios/showcases/paris_opera_multitarget/README.md)
includes its static scene and vehicle inputs.

- Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
- Paris Opéra facade and roof textures: adapted with AI assistance from
  Peter Rivera's [*Paris Opera full frontal architecture, May 2009*](https://commons.wikimedia.org/wiki/File:Paris_Opera_full_frontal_architecture,_May_2009.jpg),
  [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Changes include
  perspective correction, foreground removal, and texture preparation. See the
  [texture credit](../../scenarios/showcases/paris_opera_multitarget/inputs/scene/textures/README.md)
  for provenance and modifications.
- Eiffel Tower: 5556374, distributed by n42k through
  [3DMR](https://3dmr.eu/model/4), which declares CC0 1.0 for the download.
- Burj Khalifa: SDC PERFORMANCE / Lambo_SC04,
  [source model](https://sketchfab.com/3d-models/c1d6f5884c9c4a56b8d8f9c5555f1902),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Cars: Mike Pan and Morgan McGuire,
  [Computer Graphics Archive](https://casual-effects.com/data/), CC0/Public Domain. Model
  badges were removed for the Paris recording.
- Scooter: Innerscene,
  [Urban motor scooter](https://www.innerscene.com/tools/library/3d-parts/urban-motor-scooter-9dd945f9),
  CC0 1.0.

The walking-body model and motion inputs retain their own license terms.
Drone geometry and procedural appearance are ORCHAV assets. Environment and
surface credits are in the [third-party notices](../../THIRD_PARTY_NOTICES.md).

---

Home: [Documentation](../README.md) | Up: [Visualizer](README.md)
