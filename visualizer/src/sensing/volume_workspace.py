"""Sensing workspace navigation within the main visualizer panel."""


def reveal_sensing_panel(visualizer):
    """Expose saved volume analysis even when the scene has no top-level frames."""
    manager = visualizer.ui_manager
    section = manager.sections.get("sensing")
    if section is not None:
        section.show()
        section.expand()
    tabs = getattr(manager, "_tab_widget", None)
    page = getattr(manager, "_tab_widget_by_panel", {}).get("sensing")
    if tabs is not None and page is not None:
        index = tabs.indexOf(page)
        tabs.setTabVisible(index, True)
        tabs.setTabEnabled(index, True)
        tabs.setCurrentIndex(index)


def retire_volume_scene(visualizer):
    """Release volume UI/layers before scene destruction, except its own transition."""
    window = getattr(visualizer, "sensing_volume_window", None)
    if window is not None and not window.closed and not window.opening_scene:
        window.close()
    breadcrumb = getattr(visualizer, "volume_return", None)
    if breadcrumb is not None:
        from shiboken6 import isValid

        if isValid(breadcrumb):
            breadcrumb.close()
