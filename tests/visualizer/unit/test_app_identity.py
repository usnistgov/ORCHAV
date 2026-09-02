import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from visualizer.src.app.app_identity import (
    APPLICATION_ICON_SIZES,
    APPLICATION_NAME,
    ORGANIZATION_NAME,
    WINDOW_TITLE,
    apply_application_identity,
    apply_window_identity,
    build_orchav_icon,
)


def test_application_identity_sets_name_and_icon():
    app = QApplication.instance() or QApplication([])

    icon = apply_application_identity(app)

    assert app.applicationName() == APPLICATION_NAME
    assert app.organizationName() == ORGANIZATION_NAME
    assert not icon.isNull()
    assert not app.windowIcon().isNull()


def test_window_identity_sets_title_and_icon():
    app = QApplication.instance() or QApplication([])
    apply_application_identity(app)
    window = QWidget()

    apply_window_identity(window)

    assert window.windowTitle() == WINDOW_TITLE
    assert not window.windowIcon().isNull()


def test_build_orchav_icon_returns_icon():
    _app = QApplication.instance() or QApplication([])

    icon = build_orchav_icon()

    assert not icon.isNull()
    assert {size.width() for size in icon.availableSizes()} == set(APPLICATION_ICON_SIZES)
    for size in (16, 32, 48, 64, 128, 256):
        pixmap = icon.pixmap(size, size)
        assert pixmap.size().width() == size
        assert pixmap.size().height() == size
        assert pixmap.toImage().hasAlphaChannel()
