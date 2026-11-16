"""Lazy, exact-bin signal surfaces independent of the physical scene camera."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..app.theme import get_theme_manager
from ..widgets.wrapping_layout import WrappingLayout
from .inspection_display import PowerScale, power_db

RANGE_HALF_WIDTH = 32
DOPPLER_HALF_WIDTH = 24


class SpectrumSurface(QWidget):
    """Bounded surfaces with linked exact-bin cuts; no projected mouse picking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        controls = WrappingLayout()
        self.reset_button = QPushButton("Reset view")
        self.preset = QComboBox()
        self.preset.addItems(["Oblique", "Top-down"])
        self.scope = QComboBox()
        self.scope.addItems(["Selected neighborhood", "Whole grid", "All detections region"])
        self.scope.setToolTip(
            "Whole grid retains every recorded bin. The neighborhood zooms around the selected cell."
        )
        self.layers = (QCheckBox("Input"), QCheckBox("Threshold"))
        controls.addWidget(self.reset_button)
        controls.addWidget(self.preset)
        controls.addWidget(self.scope)
        self.scope.currentIndexChanged.connect(self._redraw)
        for layer in self.layers:
            layer.setChecked(True)
            controls.addWidget(layer)
            layer.toggled.connect(self._redraw)
        self._layout.addLayout(controls)
        self.reset_button.clicked.connect(lambda: self._camera("Oblique"))
        self.preset.currentTextChanged.connect(self._camera)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.hide()
        self._layout.addWidget(self.caption)
        self.canvas = self.axes = None
        self._key = None
        self.last_slices = None
        self._sources = ()
        self._data = None
        self._scale = PowerScale()
        self.last_probe = ()
        self._detections = ()
        self.last_report_cells = ()
        self.setMinimumSize(180, 180)
        get_theme_manager().theme_changed.connect(self._apply_theme)

    def _apply_theme(self, *_):
        """Keep this specialized 3D surface themed without rebuilding its faces."""
        if self.axes is None:
            return
        theme = get_theme_manager().current_theme
        self.canvas.figure.set_facecolor(theme.bg_primary)
        self.axes.set_facecolor(theme.bg_primary)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.label.set_color(theme.text_primary)
            axis.set_tick_params(colors=theme.text_secondary)
        legend = self.axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(theme.bg_secondary)
            for text in legend.get_texts():
                text.set_color(theme.text_primary)
        self.canvas.draw_idle()

    def _camera(self, preset: str) -> None:
        with QSignalBlocker(self.preset):
            self.preset.setCurrentText(preset)
        if self.axes is not None:
            self.axes.view_init(
                elev=90 if preset == "Top-down" else 30, azim=-90 if preset == "Top-down" else -60
            )
            self.canvas.draw_idle()

    def clear(self, reason: str = "No selected spectrum") -> None:
        self._key = self.last_slices = self._data = None
        self._sources = self.last_probe = ()
        self._detections = self.last_report_cells = ()
        self._scale.reset()
        self.caption.setText(reason)
        self.caption.setVisible(bool(reason))
        self.setToolTip(reason)
        if self.axes is not None:
            self.canvas.setToolTip(reason)
            self.axes.clear()
            self.canvas.draw_idle()

    def set_spectra(
        self,
        ranges,
        doppler,
        first,
        second,
        *,
        selected,
        labels,
        threshold=False,
        eligible=None,
        span_db=60,
        cache_key=None,
        levels=None,
        range_label="Range",
        cut="doppler",
        detections=(),
    ) -> None:
        """Draw recorded coordinates without interpolation or per-layer normalization.

        The caller supplies scientific identity and common levels; the fallback
        summary cache supports isolated callers with immutable arrays.
        """
        if min(len(ranges), len(doppler)) < 2:
            self.clear(
                "A 2D surface needs at least two bins per axis; inspect the linked 1D slice."
            )
            return
        d, r = selected
        if not (0 <= d < len(doppler) and 0 <= r < len(ranges)):
            self.clear("Selected bin is outside this recorded grid.")
            return
        identity = cache_key or (id(first), id(second), id(eligible))
        if levels is None:
            levels = self._scale.levels(
                (first, second), identity=identity, scope=(range_label, threshold), span=span_db
            )
        self._detections = tuple(sorted(set(detections)))
        self._sources = first, second, eligible
        self._data = (
            ranges,
            doppler,
            selected,
            labels,
            threshold,
            levels,
            range_label,
            cut,
            identity,
        )
        for layer, label, values in zip(self.layers, labels, (first, second), strict=True):
            layer.setText(label)
            layer.setEnabled(values is not None)
            layer.setVisible(values is not None)
        self._redraw()

    def _redraw(self, *_args) -> None:
        if self._data is None:
            return
        ranges, doppler, (d, r), labels, threshold, (bottom, top), range_label, cut, identity = (
            self._data
        )
        theme = get_theme_manager().current_theme
        visible = tuple(layer.isChecked() and layer.isEnabled() for layer in self.layers)
        key = (
            identity,
            d,
            r,
            labels,
            threshold,
            bottom,
            top,
            range_label,
            cut,
            visible,
            theme.bg_primary,
            self.scope.currentText(),
            self._detections,
        )
        if key == self._key:
            return
        self._key = key
        if self.canvas is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.canvas = FigureCanvasQTAgg(Figure())
            self._layout.addWidget(self.canvas, 1)
            self.axes = self.canvas.figure.add_subplot(111, projection="3d")
        elevation, azimuth = self.axes.elev, self.axes.azim
        self.axes.clear()
        self.axes.view_init(elev=elevation, azim=azimuth)
        if self.scope.currentText() == "Whole grid" or (
            self.scope.currentText() == "All detections region" and not self._detections
        ):
            ds, rs = slice(0, len(doppler)), slice(0, len(ranges))
        elif self.scope.currentText() == "All detections region" and self._detections:
            dd, rr = np.asarray(self._detections).T
            ds = slice(max(0, int(dd.min()) - 3), min(len(doppler), int(dd.max()) + 4))
            rs = slice(max(0, int(rr.min()) - 3), min(len(ranges), int(rr.max()) + 4))
        else:
            ds = slice(
                max(0, d - DOPPLER_HALF_WIDTH), min(len(doppler), d + DOPPLER_HALF_WIDTH + 1)
            )
            rs = slice(max(0, r - RANGE_HALF_WIDTH), min(len(ranges), r + RANGE_HALF_WIDTH + 1))
        self.last_slices = ds, rs
        x, y = np.meshgrid(ranges[rs], doppler[ds])
        first, second, eligible = self._sources
        probes, heights = [], []
        for i, (power, label, color) in enumerate(
            zip((first, second), labels, ("#3d9fec", "#ec7955"), strict=True)
        ):
            if power is None:
                continue
            raw = float(power[d, r])
            verdict = not (threshold and i == 1) or (eligible is not None and bool(eligible[d, r]))
            raw_db = float(power_db(np.asarray(raw))) if raw >= 0 else np.nan
            probes.append(
                f"{label}: {raw:.6g} linear / {raw_db:.3g} dB"
                + (" (render clipped)" if raw_db < bottom or raw_db > top else "")
                + (" (not tested / eligibility unavailable)" if not verdict else "")
            )
            if not visible[i]:
                continue
            z = np.clip(power_db(power[ds, rs]), bottom, top)
            z[~np.isfinite(power[ds, rs]) | (power[ds, rs] < 0)] = np.nan
            if threshold and i == 1:
                z[(~eligible[ds, rs]) if eligible is not None else np.ones(z.shape, bool)] = np.nan
                self.axes.plot_wireframe(
                    x, y, z, rstride=1, cstride=1, color=color, linewidth=0.7, label=label
                )
            else:
                # Explicit full quads prevent partial faces spanning unavailable bins.
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                vertices = np.stack((x, y, z), axis=-1)
                faces = np.stack(
                    (vertices[:-1, :-1], vertices[1:, :-1], vertices[1:, 1:], vertices[:-1, 1:]),
                    axis=-2,
                ).reshape(-1, 4, 3)
                faces = faces[np.isfinite(faces).all(axis=(1, 2))]
                self.axes.add_collection3d(
                    Poly3DCollection(faces, facecolor=color, alpha=0.62, linewidth=0, label=label)
                )
            # Cut orientation matches the accompanying numerical 2D slice.
            if cut == "range":
                local = r - rs.start
                if 0 <= local < x.shape[1]:
                    self.axes.plot(x[:, local], y[:, local], z[:, local], color=color, lw=2.4)
            else:
                local = d - ds.start
                if 0 <= local < x.shape[0]:
                    self.axes.plot(x[local], y[local], z[local], color=color, lw=2.4)
            if (
                verdict
                and not np.isnan(raw_db)
                and ds.start <= d < ds.stop
                and rs.start <= r < rs.stop
            ):
                height = float(np.clip(raw_db, bottom, top))
                heights.append(height)
                self.axes.scatter(
                    [ranges[r]],
                    [doppler[d]],
                    [height],
                    color=color,
                    edgecolor="#edca55",
                    s=35,
                    depthshade=False,
                )
        self.last_probe = tuple(probes)
        if threshold and len(heights) == 2:
            self.axes.plot([ranges[r]] * 2, [doppler[d]] * 2, heights, color="#edca55", lw=2)
        if eligible is not None and any(visible):
            excluded = np.where(~eligible[ds, rs], bottom, np.nan)
            self.axes.plot_wireframe(
                x, y, excluded, rstride=1, cstride=1, color="#999999", linewidth=0.7
            )
        self.last_report_cells = tuple(
            (dd, rr)
            for dd, rr in self._detections
            if ds.start <= dd < ds.stop and rs.start <= rr < rs.stop
        )
        if self.last_report_cells and first is not None and visible[0]:
            dd, rr = np.asarray(self.last_report_cells).T
            self.axes.scatter(
                ranges[rr],
                doppler[dd],
                np.clip(power_db(first[dd, rr]), bottom, top),
                color="#25bd87",
                edgecolor="#10382c",
                s=38,
                depthshade=False,
                label="Published peak cells",
            )
        self.axes.set(
            xlabel=f"{range_label} (m)",
            ylabel="Doppler (Hz)",
            zlabel="Power (dB)",
            xlim=(ranges[rs.start], ranges[rs.stop - 1]),
            ylim=(doppler[ds.start], doppler[ds.stop - 1]),
            zlim=(bottom, top + 1),
        )
        self.axes.set_box_aspect((1.4, 1, 0.7))
        self.axes.set_position((0.0, 0.08, 0.90, 0.82))
        if any(visible):
            self.axes.legend(
                loc="upper left",
                fontsize=8,
                facecolor=theme.bg_secondary,
                labelcolor=theme.text_primary,
            )
        self.canvas.figure.set_facecolor(theme.bg_primary)
        self.axes.set_facecolor(theme.bg_primary)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.label.set_color(theme.text_primary)
            axis.label.set_fontsize(9)
            axis.labelpad = 1
            axis.set_tick_params(colors=theme.text_secondary, labelsize=7)
            axis.set_pane_color((0.4, 0.4, 0.4, 0.08))
        details = (
            f"{range_label} bin {r}: {ranges[r]:.6g} m · Doppler bin {d}: {doppler[d]:.6g} Hz. "
            + " · ".join(probes)
            + f"\n{self.scope.currentText()}: bins {rs.start}–{rs.stop - 1} × {ds.start}–{ds.stop - 1}. "
            + (
                f"{len(self.last_report_cells)}/{len(self._detections)} published peak cells in view. "
                if threshold
                else ""
            )
            + f"Common floor/top {bottom:.1f}/{top:.1f} dB re 1 recorded unit; height is power, not metres. "
            + (
                "Grey floor: excluded from CFAR; no threshold surface there."
                if eligible is not None
                else "Lines connect recorded bins."
            )
        )
        self.setToolTip(details)
        self.canvas.setToolTip(details)
        self.caption.setText("Both layers hidden." if not any(visible) else "")
        self.caption.setVisible(not any(visible))
        self.canvas.draw_idle()

    def release(self) -> None:
        if self.canvas is not None:
            self.canvas.figure.clear()
            self.canvas.close()
        self.axes = self.canvas = self._key = self._data = None
        self._sources = ()
        self._detections = self.last_report_cells = ()
        self._scale.reset()
