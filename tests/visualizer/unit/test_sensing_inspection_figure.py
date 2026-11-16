"""Optional figures allocate no canvas until requested and release on collapse."""

from visualizer.src.sensing.inspection_figure import InspectionFigurePanel


def test_inspection_figure_is_lazy_and_replaces_stale_content(qtbot):
    from matplotlib.figure import Figure

    panel = InspectionFigurePanel("Details")
    qtbot.addWidget(panel)
    calls = []

    def build():
        calls.append(True)
        return Figure()

    panel.set_source(build)
    assert panel.canvas is None and calls == []
    panel.toggle.setChecked(True)
    assert panel.canvas is not None and len(calls) == 1
    panel.set_source(None)
    assert panel.canvas is None
    assert "not recorded" in panel.note.text()
    panel.toggle.setChecked(False)
    panel.set_source(build)
    assert len(calls) == 1
    panel.toggle.setChecked(True)
    assert len(calls) == 2
    panel.toggle.setChecked(False)
    assert panel.canvas is None


def test_stable_source_and_reset_release_canvas(qtbot):
    from matplotlib.figure import Figure

    panel = InspectionFigurePanel("Details")
    qtbot.addWidget(panel)
    panel.set_source(Figure, key="event-one")
    panel.toggle.setChecked(True)
    canvas = panel.canvas
    panel.set_source(Figure, key="event-one")
    assert panel.canvas is canvas
    panel.reset()
    assert panel.canvas is None and panel.factory is None
    assert not panel.toggle.isChecked()


def test_inspection_legend_is_outside_axes_and_inside_figure(qtbot):
    from matplotlib.figure import Figure

    panel = InspectionFigurePanel("Details")
    qtbot.addWidget(panel)
    panel.resize(700, 600)

    def build():
        figure = Figure(layout="constrained")
        axis = figure.subplots()
        axis.plot([0, 1], [1, 2], label="Saved detector input")
        axis.plot([0, 1], [0, 1], label="Saved threshold")
        axis.legend()
        return figure

    panel.set_source(build)
    panel.show()
    panel.toggle.setChecked(True)
    panel.canvas.draw()
    figure = panel.canvas.figure
    axis = figure.axes[0]
    renderer = panel.canvas.get_renderer()
    legend = axis.get_legend().get_window_extent(renderer)
    assert not legend.overlaps(axis.get_window_extent(renderer))
    assert figure.bbox.contains(*legend.min) and figure.bbox.contains(*legend.max)


def test_report_retirement_does_not_reenter_dashboard_callbacks(qtbot):
    from matplotlib.figure import Figure

    from visualizer.src.sensing.sensing_window import SensingWindow

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window.association_figure.set_source(Figure)
    window.association_figure.toggle.setChecked(True)
    window.close()
    assert window.association_figure.canvas is None
    assert window.truth_zoom.dashboard is None
    window._rerender_last_request()


def test_neutral_report_text_theme_roundtrip_preserves_semantic_colors(qtbot):
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    from visualizer.src.app.theme import current_theme, get_theme_manager

    panel = InspectionFigurePanel("Saved report")
    qtbot.addWidget(panel)

    def build():
        figure = Figure()
        axis = figure.subplots()
        figure.suptitle("Saved heading")
        axis.annotate("D0", (0, 0))
        axis.annotate("Track", (1, 1), color="C2")
        return figure

    panel.set_source(build)
    panel.toggle.setChecked(True)
    manager = get_theme_manager()
    # Apply both actual theme palettes without rebuilding the figure.
    old = manager.mode
    figure = panel.canvas.figure
    try:
        for mode in ("dark", "light", "dark"):
            manager.set_mode(mode, persist=False)
            panel._apply_theme()
            assert to_rgba(figure._suptitle.get_color()) == to_rgba(current_theme().text_primary)
            assert to_rgba(figure.axes[0].texts[0].get_color()) == to_rgba(
                current_theme().text_primary
            )
            assert figure.axes[0].texts[1].get_color() == "C2"
            assert panel.canvas.figure is figure
    finally:
        manager.set_mode(old, persist=False)
