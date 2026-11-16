"""The contextual map ignores graph results from an abandoned document."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem

from visualizer.src.authoring.trajectory_preview import TrajectoryPreview


def map_asset(texture):
    return SimpleNamespace(
        attribution="© OpenStreetMap contributors",
        material=SimpleNamespace(texture_path=str(texture)),
        mesh=SimpleNamespace(
            vertices=np.array(((0, 0, 0), (16, 0, 0), (0, 16, 0))),
            triangle_uvs=np.array(((0, 0), (1, 0), (0, 1))),
        ),
    )


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_saved_map_png_preserves_pixels_alpha_and_context_cache(qtbot, tmp_path, mode):
    texture = tmp_path / "map.png"
    color = (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60)
    Image.new(mode, (16, 16), color).save(texture)
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    preview.set_context((map_asset(texture),), key="map")
    item = preview._background[0]
    assert isinstance(item, QGraphicsPixmapItem)
    expected = (*color[:3], color[3] if mode == "RGBA" else 255)
    assert item.pixmap().toImage().pixelColor(0, 0).getRgb() == expected
    assert item.opacity() == 0.65 and item.zValue() == -2
    assert "OpenStreetMap" in preview._map_credit
    preview.set_context((map_asset(texture),), key="map")
    assert preview._background == [item]


def test_changed_png_context_bypasses_qt_filename_cache(qtbot, tmp_path):
    texture = tmp_path / "map.png"
    Image.new("RGB", (16, 16), "red").save(texture, compress_level=0)
    original_stat = texture.stat()
    assert QPixmap(str(texture)).toImage().pixelColor(0, 0).name() == "#ff0000"
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    asset = map_asset(texture)
    preview.set_context((asset,), key="red")

    Image.new("RGB", (16, 16), "blue").save(texture, compress_level=0)
    assert texture.stat().st_size == original_stat.st_size
    os.utime(texture, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    preview.set_context((asset,), key="blue")
    assert preview._background[0].pixmap().toImage().pixelColor(0, 0).name() == "#0000ff"


@pytest.mark.parametrize(
    "payload",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect width="16" height="16" fill="green"/></svg>',
        b"not a PNG",
    ],
    ids=["renamed_svg", "corrupt"],
)
def test_changed_map_is_rejected_without_stale_pixmap_or_losing_streets(qtbot, tmp_path, payload):
    texture = tmp_path / "map.png"
    Image.new("RGB", (16, 16), "green").save(texture)
    asset = map_asset(texture)
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    preview.set_context((asset,), key="valid")
    assert len(preview._background) == 1
    # Populate Qt's automatic filename cache, then change the image after
    # asset preparation. A new preview context must inspect the new bytes.
    assert not QPixmap(str(texture)).isNull()
    original_stat = texture.stat()
    texture.write_bytes(payload)
    os.utime(texture, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    preview.set_context((asset,), streets=(((0, 0, 0), (1, 1, 0)),), key="invalid")
    assert len(preview._background) == 1
    assert not any(isinstance(item, QGraphicsPixmapItem) for item in preview._background)
    assert not preview._map_credit
    preview.set_paths(((0, 0, 0), (1, 1, 0)))
    assert preview.committed == ((0, 0, 0), (1, 1, 0))

    Image.new("RGB", (16, 16), "blue").save(texture)
    preview.set_context((asset,), key="repaired")
    assert preview._background[0].pixmap().toImage().pixelColor(0, 0).name() == "#0000ff"


def test_stale_street_results_do_not_replace_current_context(qapp):
    preview = TrajectoryPreview()
    preview.set_context(key="old")
    preview.set_context(key="current")
    preview.set_paths(((1, 2, 3), (4, 5, 6)), current=(1, 2, 3))
    before = len(preview.canvas.items())
    preview._show_streets("old", (((0, 0, 0), (10, 10, 0)),))
    assert len(preview.canvas.items()) == before
    preview._show_streets("current", (((0, 0, 0), (10, 10, 0)),))
    assert len(preview.canvas.items()) == before + 1
    assert preview.committed == ((1, 2, 3), (4, 5, 6))
    preview.close()


def test_wheel_zoom_keeps_pointer_location_and_survives_playback(qapp, qtbot):
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    preview.resize(360, 500)
    preview.show()
    path = ((10, 20, 3), (40, 50, 6))
    preview.set_paths(path, current=path[0])
    qapp.processEvents()
    view = preview.view
    anchor = view.viewport().rect().center() + QPoint(25, -20)
    before = view.mapToScene(anchor)
    scale = view.transform().m11()
    event = QWheelEvent(
        QPointF(anchor),
        QPointF(view.viewport().mapToGlobal(anchor)),
        QPoint(),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    qapp.sendEvent(view.viewport(), event)
    assert view.transform().m11() == pytest.approx(scale * 1.2)
    assert not preview.auto_fit.isChecked()
    after = view.mapToScene(anchor)
    assert after.x() == pytest.approx(before.x(), abs=1 / scale)
    assert after.y() == pytest.approx(before.y(), abs=1 / scale)
    manual_transform = view.transform()
    preview.set_paths(path, current=path[1])
    preview.resize(420, 540)
    qapp.processEvents()
    assert view.transform() == manual_transform


def test_auto_fit_frames_paths_instead_of_entire_map(qapp, qtbot):
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    preview.resize(360, 500)
    preview.show()
    preview.set_context(streets=(((-10000, -10000, 0), (10000, 10000, 0)),), key="city")
    preview.set_paths(((10, 20, 3), (40, 50, 6)))
    qapp.processEvents()
    view = preview.view
    automatic_scale = view.transform().m11()
    assert view.mapToScene(view.viewport().rect()).boundingRect().width() < 100
    preview.zoom_in_button.click()
    assert view.transform().m11() > automatic_scale
    preview.zoom_out_button.click()
    assert view.transform().m11() == pytest.approx(automatic_scale)
    assert not preview.auto_fit.isChecked()
    preview.set_paths(((200, 300, 1), (240, 350, 5)))
    preview.fit_button.click()
    assert view.viewport().rect().contains(view.mapFromScene(QPointF(220, -325)))
    assert not preview.auto_fit.isChecked()
    preview.auto_fit.setChecked(True)
    preview.set_paths(((500, 600, 1), (800, 900, 5)))
    assert view.viewport().rect().contains(view.mapFromScene(QPointF(650, -750)))
    assert view.transform().m11() < automatic_scale


def test_current_marker_outline_stays_thin_when_fitting_a_short_walk(qapp, qtbot):
    preview = TrajectoryPreview()
    qtbot.addWidget(preview)
    preview.resize(360, 500)
    preview.show()
    path = ((-49.59, -56.75, 1.04), (-48.34, -56.42, 1.05))
    preview.set_paths(path, current=path[0])
    qapp.processEvents()
    marker = next(
        item
        for item in preview.canvas.items()
        if isinstance(item, QGraphicsEllipseItem) and item.brush().color().name() == "#ffffff"
    )
    assert preview.view.transform().m11() > 10
    # A world-space 1 m outline obscures this entire short trajectory.
    assert marker.pen().isCosmetic()
