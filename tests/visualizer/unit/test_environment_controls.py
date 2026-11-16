"""Optional-sky UI: selection, failure, cancellation and ordinary renderer wiring."""

from threading import Event
from types import SimpleNamespace

from visualizer.src.panels import environment_download as ui
from visualizer.src.panels.render_panel import RenderPanel
from visualizer.src.renderers.protocol import RendererCapabilities


def test_missing_environment_is_explicit_and_cancel_safe(qtbot, monkeypatch):
    monkeypatch.setattr(ui, "cached_environment", lambda _: None)
    started = Event()
    stopped = Event()

    def download(asset, *, cancelled, progress):
        started.set()
        assert cancelled.wait(2)
        stopped.set()
        raise ui.DownloadCancelled()

    monkeypatch.setattr(ui, "download_environment", download)
    widget = ui.EnvironmentDownload()
    qtbot.addWidget(widget)
    received = []
    widget.ready.connect(received.append)
    widget.select("sky_sunny")
    assert not started.is_set()
    assert "4.4 MiB" in widget.download.text()
    widget.start_download()
    qtbot.waitUntil(started.is_set)
    widget.select("sky_overcast")
    qtbot.waitUntil(stopped.is_set)
    assert widget.selected.id == "sky_overcast"
    assert not received
    widget.cleanup()


def test_download_failure_is_retryable_and_does_not_apply(qtbot, monkeypatch):
    monkeypatch.setattr(ui, "cached_environment", lambda _: None)

    def fail(*a, **kw):
        raise OSError("offline")

    monkeypatch.setattr(ui, "download_environment", fail)
    widget = ui.EnvironmentDownload()
    qtbot.addWidget(widget)
    received = []
    widget.ready.connect(received.append)
    widget.select("sky_sunset")
    widget.start_download()
    qtbot.waitUntil(lambda: "offline" in widget.status.text())
    assert widget.download.isEnabled()
    assert not received
    widget.cleanup()


def test_success_applies_only_after_worker_completion(qtbot, monkeypatch):
    present = set()
    monkeypatch.setattr(ui, "cached_environment", lambda a: a.id if a.id in present else None)

    def download(asset, **kwargs):
        present.add(asset.id)

    monkeypatch.setattr(ui, "download_environment", download)
    widget = ui.EnvironmentDownload()
    qtbot.addWidget(widget)
    received = []
    widget.ready.connect(received.append)
    widget.select("sky_cloudy")
    assert received == []
    widget.start_download()
    qtbot.waitUntil(lambda: received == ["sky_cloudy"])
    assert not widget.awaiting_download
    widget.cleanup()


def test_normal_panel_has_only_default_and_four_optional_skies(qtbot, monkeypatch):
    monkeypatch.setattr(ui, "cached_environment", lambda _: None)
    calls = []
    current = {"ibl": "neutral_outdoor"}

    def set_ibl(name):
        calls.append(name)
        current["ibl"] = name
        return True

    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(ibl=True, hdr_ibl=True, skybox=True),
        set_ibl=set_ibl,
        get_ibl_name=lambda: current["ibl"],
        get_ibl_intensity=lambda: 6000,
        apply_environment_lighting=lambda name: calls.append(("recipe", name)),
        show_skybox=lambda value: calls.append(("skybox", value)),
    )
    panel = RenderPanel(SimpleNamespace(renderer=renderer))
    group = panel.create_lighting_panel()
    qtbot.addWidget(group)
    combo = panel.widgets["ibl_combo"]
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Default",
        "Overcast",
        "Sunny",
        "Partly cloudy",
        "Sunset",
    ]
    combo.setCurrentText("Sunny")
    assert calls == []
    panel._sync_ibl_selection("neutral_outdoor")
    assert combo.currentText() == "Sunny"  # download action remains accessible
    monkeypatch.setattr(ui, "cached_environment", lambda _: "cached.hdr")
    combo.setCurrentText("Overcast")
    assert "sky_overcast" in calls and ("recipe", "sky_overcast") in calls
    panel.widgets["environment_download"].match_lighting.setChecked(False)
    calls.clear()
    combo.setCurrentText("Sunset")
    assert "sky_sunset" in calls and not any(
        isinstance(c, tuple) and c[0] == "recipe" for c in calls
    )
    calls.clear()
    combo.setCurrentText("Default")
    assert calls == ["neutral_outdoor"]
    assert combo.currentText() == "Default"
    assert current["ibl"] == "neutral_outdoor"
    panel.cleanup()
