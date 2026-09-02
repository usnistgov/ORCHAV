# MPC Explorer

The MPC Explorer is a resizable table and selected-path view for classifying
every [multipath component
(MPC)](../concepts/propagation.md#multipath-components) in the displayed frame.
It complements the scene-wide controls in **Paths**: that panel controls the
bulk path population and drawing, while the Explorer identifies individual
paths, their ordering, interactions, and materials.

Open it with **Open MPC Explorer...** beside the MPC count in **Paths**.

Use the [MPC Inspection
scenario](../../scenarios/visualizer/mpc_inspection/README.md) to explore path
classification and selection with a generated frame. Its scenario page
provides the generation and launch commands.

## Inspect a path

1. Choose the population with **Scope**.
2. Choose one complete recipe under **Preset recipes**, or switch to **Custom
   order** to choose a group and author the compound order yourself.
3. Open **Filters** to narrow the result.
4. Select a table row. With the pygfx renderer, the selected path receives a
   halo, a crisp foreground line, an arrow, a looping flow pulse, colored
   bounce markers, and numbered bounce labels.
5. Read the complete interaction and material sequences in **Selected path
   details**.

A short, unmodified left click on a visible pygfx MPC selects the same
frame-local path and synchronizes the table when the Explorer is open. Camera
drags do not select a path, and viewport path selection is disabled while an
authoring or actor-editing session owns the viewport.

The pulse is a TX-to-RX direction cue, not a representation of physical
propagation speed.

## Scopes and status

The count below the table shows matching paths over the complete path set in
the frame, for example `304,437 / 2,000,000 paths`.

| Scope | Population |
|-------|------------|
| **All paths** | Every path in the displayed frame. |
| **Filtered paths** | Paths that passed the Paths-panel filters before the optional Top-K limit. |
| **Actually rendered paths** | Paths with at least one segment in the final bulk MPC layer after filtering, Top-K, and effective path visibility. |

Selecting a path from **All paths** or **Filtered paths** always uses its
complete path geometry. The selected-path overlay can therefore show a path
that is outside the current rendered set, or show the complete
mixed-mechanism path when only some of its segments pass a bulk segment filter.
The status column and detail pane identify these cases. Selection never
recolors or replaces the bulk MPC geometry.

Path IDs are zero-based and frame-local. A successfully presented
replacement frame clears the selection. Sorting and filtering within the same
frame preserve it by path ID. The Explorer does not attempt cross-frame path
tracking.

## Table columns

The default table contains:

- Path ID
- TX and RX IDs
- Whole-path loss in dB
- Whole-path delay in ns
- Interaction count
- Interaction mix
- First bounce material
- Filtered/rendered status

Use **Columns** for optional angles, geometric length, stretch ratio,
pair-relative metrics, provenance, and complete interaction or material
sequences. Angles that were not exported appear as unavailable rather than
`0 deg`. A measured zero-degree angle remains a real value. Long sequence
strings are prepared on demand for visible cells and the selected-path detail
pane.

Angles use world-space degrees. Azimuth is in `[0, 360)`, with `+X = 0 deg`
and `+Y = 90 deg`. Elevation is in `[-90, 90]`, with positive values pointing
toward `+Z`. AoD describes the first segment leaving TX, and AoA describes the
final segment arriving at RX. For a LoS path their azimuths normally differ by
180 degrees. Azimuth is circular, so numeric sorting crosses the seam between
359 and 0 degrees even though those directions are adjacent.

"Interactions" counts all nonzero mechanisms along the path. It is not
"reflection order": diffuse scattering, refraction, diffraction, and other
mechanisms count as interactions too. Only whole-path loss and delay are
available. The Explorer does not invent per-bounce RF values.

## Grouping and ordering

The two ordering tabs are mutually exclusive:

- **Preset recipes** contains complete one-click classifications. A preset
  chooses both its grouping and its within-group order. The **Group rows by**
  control does not modify it.
- **Custom order** exposes the grouping and two to four explicit sort clauses.
  Changes are labeled as a draft until **Apply custom** is pressed.

The preset recipes mean:

| Preset | Result |
|--------|--------|
| **Per TX/RX — strongest first** | Separate each pair, then order by lowest path loss. |
| **Per TX/RX — earliest first** | Separate each pair, then order by lowest delay. |
| **Global — strongest first** | One ungrouped path-loss ordering. |
| **Global — earliest first** | One ungrouped delay ordering. |
| **By interaction count — strongest first** | Separate bounce counts, strongest within each count. |
| **By interaction mix — strongest first** | Separate mechanism mixes, strongest within each mix. |
| **By first material — strongest first** | Separate first-bounce materials, strongest within each material. |
| **By delay band — strongest first** | Separate 10 ns delay bands, strongest within each band. |
| **By path-loss band — earliest first** | Separate 10 dB loss bands, earliest within each band. |

Band presets first classify paths into buckets, then apply the named ordering
inside each bucket. They are therefore different from a global ordering.

Custom grouping remains a flat table with a light separator when the group
changes. Available groups are:

- None (global order)
- TX -> RX
- RX -> TX
- Interactions
- Interaction mix
- First material
- Delay band
- Path-loss band

The default recipe orders TX ascending, RX ascending, path loss ascending,
delay ascending, then path ID ascending. Lower path loss means a
stronger path. Path ID is always the final deterministic tie-breaker even
though it does not consume a custom clause.

In **Custom order**, group chips are locked at the front. The first ordinary
clause is the primary within-group order. Later clauses only resolve ties.
**Use as primary** moves the selected field to the front, while **Add clause**
appends a lower-priority tie-breaker.

Clicking a table header enters **Custom order**, uses global grouping, and makes
that column primary. Clicking again reverses the direction. With pair grouping,
an angle or metric restarts inside each pair rather than remaining monotonic
across the complete table. Missing or non-finite values sort after available
values.

Delay and path-loss grouping use 10 ns and 10 dB bands by default. A band is a
classification aid. It does not change the underlying metric.

## Filters and mechanism classification

The filter drawer accepts comma-separated TX IDs, RX IDs, mechanism IDs, and
first-material IDs, plus unrestricted numeric ranges for path loss, delay, and
interaction count. The interaction IDs are `0` for LoS, `1` for specular
reflection, `2` for diffuse scattering, `4` for refraction, `8` for
diffraction, and `99` for a virtual bounce in a reconstructed path. Material
IDs are frame-local. Use the table or detail pane to relate an ID to its
material name.

Mechanism filters have distinct meanings:

- **Contains mechanisms** uses AND semantics. A path must contain every listed
  mechanism somewhere in its sequence. Code `0` matches LoS paths.
- **Pure mechanism** matches paths whose nonzero interactions use only the
  specified mechanism.
- **Mixed mechanisms only** matches paths with more than one nonzero mechanism
  category.
- **Exact sequence** matches the ordered interior interaction sequence.

The Explorer does not reduce a path to one scalar "type." A path such as
specular, diffuse, specular is both mixed and sequence-sensitive.

## Pair-relative classifications

Pair-relative values are computed independently inside each TX/RX pair:

- **Pair strength rank** orders finite path loss from lowest to highest, with
  path ID as the deterministic tie-breaker.
- **Delta loss from strongest** subtracts the pair's lowest finite path loss.
- **Excess delay** subtracts the pair's earliest finite delay.
- **Relative power proxy** is the dimensionless quantity
  `10 ** (-path_loss_db / 10)`.

The proxy is neither received dBm nor absolute received power. It does not
include transmit power, antenna gain, phase, receiver response, or any other
link-budget term. For cumulative contribution classification, proxies are
normalized only within one TX/RX pair and accumulated from strongest to
weakest to define the minimal 50%, 90%, and 99% sets. Missing path loss or delay
produces an unavailable pair-relative value.

## Metric provenance

An exported whole-path delay or path loss is labeled **exported**. When an
input lacks one of those values, ORCHAV may supply a geometric estimate and
label it **estimated**. A supplied value without provenance is labeled
**unknown** rather than being described as measured.

The estimated/unknown label applies to the whole-path metric. It does not
create per-segment or per-bounce delay, loss, power, phase, or deposited-energy
measurements.

## Large frames

The Explorer builds its path catalog only while it is open. Filtering and
sorting very large frames can take additional time. Closing the Explorer stops
its queries and removes the selected-path overlay.

## Renderer Support and Limits

The table and details pane work with both renderers. The halo, pulse, direction
cue, bounce markers, labels, and viewport picking are pygfx-only. Open3D users
can still classify and inspect paths from the table.

The Explorer does not provide:

- Exact path identity across frames.
- Per-bounce path loss, delay, phase, or energy.
- Building, facet, or triangle identity for a bounce.
- Fresnel zones or RF hotspot overlays.

See [Multipath Components](../concepts/propagation.md#multipath-components),
[Interaction Types](../concepts/propagation.md#interaction-types), and
[Path Metrics](../concepts/propagation.md#path-metrics) for the underlying propagation
concepts.

---

Up: [Visualizer](README.md) | Related: [Visual Analysis](analysis.md) | [Beam-Pattern Visualization](beam_patterns.md)
