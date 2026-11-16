# Documentation Style

Use these conventions for new pages and for sections changed in a contribution.
They define the current publishing standard, not a claim that every untouched
historical page already has the same structure or level of detail.

## Change Scope

Changed documentation must remain technically accurate, keep relative links
and anchors valid, follow the Mermaid rules below, and respect release-surface
boundaries. Regenerate and check generated references when their source changes.

Across the full documentation set, progressive disclosure, canonical ownership,
and consistent navigation are ongoing goals. Apply them fully to new pages and
substantial rewrites. For a small correction, keep the change local unless an
untouched passage would leave a contradiction or broken route. This scope does
not relax pull-request or release checks.

Progressive disclosure introduces the normal workflow first, links to focused
guides when a decision appears, and reserves exact field and API details for
reference pages.

## Accessible, Progressive Explanations

Write for technically proficient readers who may be new to ORCHAV's
specific interfaces and conventions. State the capability, its prerequisites
and the relevant workflow directly, then define unfamiliar terms where
necessary. Use concise, precise language without assuming a lack of general
technical knowledge.

Maintain a neutral, professional tone throughout public documentation.
Avoid promotional hooks, rhetorical sales questions, reassurance-oriented
phrases and emphatic claims that a task is "easy" or "simple". Use bold
type for genuine distinctions or warnings, not to persuade the reader.
Prefer factual descriptions such as "Directory-backed PLY sequences do not
require a timed manifest" over conversational questions or encouragement.

Keep the repository README focused on product purpose, component roles, the
frame workflow, and a first run. Preserve the producer, shared storage/provider,
and consumer roles, but put field lists, transport restrictions, and processing
details in the linked guides and references. Include a qualification on the
landing page only when omitting it would mislead the reader about that workflow.

Keep overview diagrams compact and readable. Use short labels and let the
surrounding prose explain them. Avoid repeating a field inventory or the same
workflow in both the figure and several paragraphs. Examples links should say
what users can adapt or accomplish, without listing every example.

Keep scenario catalogs short: state what each example teaches and link to it.
On a walkthrough page, include the complete preparation and launch sequence,
then ordered interface steps with the control names and expected result. Links
should deepen the explanation, not supply a missing step in the normal run.
Put connection overrides, stress workloads, and implementation details after
the first successful run or in the relevant reference page.

## Page purpose

A reader should recognize from a page's opening what it explains, when it
is useful, and how it differs from neighboring guides or runnable examples.
This does **not** require a literal "Goal" heading on every page. For
specialized workflows, distinguish the general capability from optional inputs
and application-specific examples.

Make the promise of each navigation link match its destination. A link labelled
"inspect detector decisions" should lead to the available plots, tables,
controls, saved-evidence prerequisites and a runnable example, not only to
an abstract algorithm description or one illustration. Put reusable processing
capabilities in a focused guide, exact settings in a reference, and the
steps/results of an experiment in its scenario.

Before adding a page, identify its reader, the task or question it answers, and
where the reader arrives from. Extend an existing guide when it already owns
that task. Keep the four component landing pages as architectural introductions
with an explained figure, then route readers to guides, examples and reference.

A guide that generates or exports a figure should show a representative result
and explain what to observe. Use an existing scenario where possible. Reference
field tables and API contracts do not need decorative images.

Historical test counts, capture-harness notes and qualification reports belong
in maintainer evidence. They are not a substitute for current user instructions.

## Page Ownership

Use one canonical owner for each subject. Other pages should give only enough
context to make their link understandable. Use this map when adding or
substantially revising material.

| Subject | Preferred canonical owner |
| --- | --- |
| Product identity and component roles | Repository `README.md` |
| Documentation routes | `docs/README.md` |
| First end-to-end run | Getting Started Quickstart |
| Shared cross-component terms | Glossary |
| Scenario creation | Generator Scenario Authoring |
| Exact YAML fields | Scenario YAML Reference |
| Frame flow and data-mode choice | Shared Data Layer guide |
| Visualizer operation | Visualizer guides |
| Runnable values and observations | The relevant scenario README |
| Internal implementation seams | Developer Architecture |

Use **Generator** for ORCHAV's primary frame producer, **Shared Data Layer** for
the contracts, storage, transport codecs, and frame providers, and
**Visualizer** for the primary interactive frame consumer. Use *storage* for
HDF5 and *transport* for protobuf over gRPC.

Use *multipath components (MPCs)* or *propagation paths* when discussing the
ray-traced paths, and *complex channel samples* or *channel observations* for
the sensing processing inputs. Name processed outputs directly, such as range
profiles, detections, or tracks.

Use *frame source* for the place or service from which frames originate. Use
*frame provider* for the software interface that retrieves frames from one
delivery route for a consumer. Do not use *frame reader* as a synonym for a
frame provider. Reserve *reader* for a component that directly parses a file
format.

## Terms And Links

The [Glossary](../reference/glossary.md) is an optional lookup, not required
reading. Define a core term briefly at its first meaningful use on a landing or
introductory page, even when a glossary entry exists. Link the term itself in
the normal flow of the sentence, for example:

```markdown
A [frame](../reference/glossary.md#frame) is one snapshot of a scenario step.
```

Do not link every repetition. Each global glossary term has its own `###`
heading.
Within a definition, link mentions of other glossary terms so a new reader can
follow unfamiliar concepts. End each entry with a normal link to its canonical
workflow, concept, or reference page. A focused page may keep a small local
glossary for abbreviations used only there, but it should not redefine shared
cross-component terms.

For other links, name the destination or the question it answers. Avoid vague
labels such as *here* or *more*. Link to an exact section when that section,
rather than the whole page, answers the reader's question.

Prefer short sentences or parentheses to em dashes and semicolons in public
prose. Keep ordinary hyphens in compound terms, file names, and command-line
options.

## Mermaid Diagrams

- Use `flowchart TB` and top-to-bottom subgraphs for architecture and workflow
  diagrams.
- Prefer separate nodes and labeled edges to HTML inside a node label.
- Do not use `<br>` in Mermaid labels. It can collapse words in some renderers,
  accessibility views, and plain-text extraction.
- A compact side-by-side comparison may use `flowchart LR` only when each
  compared flow remains vertical and the Mermaid block contains:

  ```text
  %% orchav-docs: allow-horizontal-comparison
  ```

- Inspect the rendered result whenever a diagram changes. Confirm that labels
  have visible word boundaries and that component ownership is unambiguous.

Run the automated source check from the repository root:

```bash
python scripts/ci/check_documentation.py
```

The public export audit applies the same rules to the generated documentation
surface.

## Navigation

When adding or changing navigation, use these labels consistently:

- **Home** returns to the repository or documentation landing page. **Up**
  returns to the immediate section or scenario collection.
- **Begin** moves from a collection page to the first step in one clearly
  identified path.
- **Previous** and **Next** are reciprocal and appear only within one genuine,
  named sequence or learning track.
- **Continue** is a one-way handoff from the end of one path to another area.
  It does not imply a reciprocal **Previous** link.
- **Related** lists unordered contextual links. Reciprocal **Related** links
  are acceptable because they do not define a reading sequence.
- Reference pages are terminal lookup nodes. Give them **Home** and only the
  few **Related** links needed to interpret the reference. Do not add a
  backlink to every guide that cites them.
- Task routers use headings such as **Choose A Task** or **Related Tasks**.
- Task-navigation tables use **Goal** as their first column heading.
  Reserve **Continue** for a genuine forward handoff.

## Demo Videos

Keep GitHub's inline video players as the primary way to watch the demos, with
a bundled MP4 link below each player as a backup. Preserve the bare attachment
URLs that GitHub renders as players. An automated request failure alone is not
a reason to replace a player with a link or poster. Check rendered playback
when possible and report any access limitation separately.

Keep source credits near the videos and link the asset's attribution record
for detailed provenance and modifications.

## Validation

For documentation-only changes, run the documentation check on the touched
paths, validate their relative links and anchors, and run `git diff --check`:

```bash
python scripts/ci/check_documentation.py path/to/changed-page.md
git diff --check -- path/to/changed-page.md
```

Run focused generator or documentation tests for any generated page. When a
code example, schema, scenario, or generated asset changes, also run the checks
for that surface.

Before a pull request, follow the full smoke-test requirements in
[Contributing](../../CONTRIBUTING.md#checks). A generated release candidate must
also pass the exporter's complete link and anchor checks and the release audit.

Home: [Documentation](../README.md) | Related: [Developer Architecture](architecture.md) |
[Contributing](../../CONTRIBUTING.md)
