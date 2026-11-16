"""Street names and route previews instead of manual graph-node lookup."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .street_routes import StreetNetwork
from .trajectory_preview import TrajectoryPreview


class StreetPlacementDialog(QDialog):
    """Preview a reproducible batch before changing the document."""

    def __init__(self, graph_path, catalog_ids, parent=None, *, selected_actor=None):
        super().__init__(parent)
        self.setWindowTitle("Place actors on streets")
        self.resize(820, 720)
        self.routes = ()
        self.network = None
        self.selected_actor = selected_actor
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose connected streets, preview the routes, then add the actors. Saved road data is used offline."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)
        graph_row = QWidget()
        row = QHBoxLayout(graph_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.graph_path = QLineEdit(str(graph_path or ""))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        reload = QPushButton("Load graph")
        reload.clicked.connect(self.load_graph)
        row.addWidget(self.graph_path, 1)
        row.addWidget(browse)
        row.addWidget(reload)
        form.addRow("Street graph", graph_row)
        self.action = QComboBox()
        self.action.addItem("Add new targets", "add")
        if selected_actor is not None:
            self.action.addItem(f"Set route for {selected_actor.name}", "selected")
        form.addRow("Action", self.action)
        self.travel_mode = QComboBox()
        for label, value in (
            ("Vehicles", "car"),
            ("Pedestrians", "pedestrian"),
            ("Bicycles", "bike"),
        ):
            self.travel_mode.addItem(label, value)
        form.addRow("Travel mode", self.travel_mode)
        self.start_street, self.end_street = QComboBox(), QComboBox()
        for combo in (self.start_street, self.end_street):
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.completer().setFilterMode(Qt.MatchContains)
        form.addRow("Start street", self.start_street)
        form.addRow("End street", self.end_street)
        self.count = QSpinBox()
        self.count.setRange(1, 100)
        self.count.setValue(5)
        form.addRow("Actors", self.count)
        self.asset = QComboBox()
        for value in catalog_ids:
            self.asset.addItem(value, value)
        form.addRow("Target appearance", self.asset)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.01, 100)
        self.speed.setValue(8)
        self.speed.setSuffix(" m/s")
        form.addRow("Speed", self.speed)
        self.altitude = QDoubleSpinBox()
        self.altitude.setRange(-10000, 100000)
        self.altitude.setValue(0.9)
        self.altitude.setSuffix(" m")
        self.altitude.setToolTip(
            "Actor origin in world Z, plus any bridge height stored in the graph. Use Placement checks to inspect ground clearance."
        )
        form.addRow("Origin altitude", self.altitude)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(1)
        form.addRow("Seed", self.seed)
        self.note = QLabel(
            "Routes follow mapped edges. Pedestrians use walkable roads and paths; sidewalks are used only where mapped. Actors hold at the route end."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.preview_map = TrajectoryPreview(self)
        self.view = self.preview_map.view
        self.canvas = self.preview_map.canvas
        layout.addWidget(self.preview_map, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.preview = buttons.addButton("Preview routes", QDialogButtonBox.ActionRole)
        self.apply = buttons.addButton("Add actors", QDialogButtonBox.AcceptRole)
        self.apply.setEnabled(False)
        self.preview.clicked.connect(self.preview_routes)
        self.apply.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.action.currentIndexChanged.connect(self._action_changed)
        self.travel_mode.currentIndexChanged.connect(self._mode_changed)
        self.graph_path.textChanged.connect(self._invalidate_graph)
        for combo in (self.start_street, self.end_street, self.asset):
            combo.currentTextChanged.connect(self.invalidate)
        for spin in (self.count, self.speed, self.altitude, self.seed):
            spin.valueChanged.connect(self.invalidate)
        self._mode_changed()

    def invalidate(self, *_):
        self.routes = ()
        self.apply.setEnabled(False)

    def _invalidate_graph(self, *_):
        self.network = None
        self.invalidate()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Saved street graph", self.graph_path.text(), "Graphs (*.graphml *.xml *.json)"
        )
        if path:
            self.graph_path.setText(path)
            self.load_graph()

    def _action_changed(self):
        adding = self.action.currentData() == "add"
        self.count.setEnabled(adding)
        self.asset.setEnabled(adding)
        self.apply.setText("Add actors" if adding else "Preview in actor editor")
        self.invalidate()

    def _mode_changed(self):
        mode = self.travel_mode.currentData()
        preferred = "procedural_car" if mode == "car" else "nist_human_walking"
        index = self.asset.findData(preferred)
        if index >= 0:
            self.asset.setCurrentIndex(index)
        self.speed.setValue({"car": 8, "pedestrian": 1.4, "bike": 4}[mode])
        self.load_graph()

    def load_graph(self):
        self.invalidate()
        self.network = None
        self.preview_map.set_context(key=object())
        self.preview_map.set_paths()
        try:
            self.network = StreetNetwork(
                Path(self.graph_path.text()), self.travel_mode.currentData()
            )
            for combo in (self.start_street, self.end_street):
                old = combo.currentText()
                combo.clear()
                combo.addItem("Any connected street", None)
                for name in sorted(self.network.streets, key=str.casefold):
                    combo.addItem(name, name)
                index = combo.findText(old)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self._draw()
            self.status.setText(
                f"{len(self.network.streets)} named streets · {self.network.graph.number_of_nodes()} junctions. Grey: usable streets; preview colours: selected routes."
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText(f"Cannot load this graph: {exc}")

    @staticmethod
    def _street(combo):
        index = combo.findText(combo.currentText())
        if index < 0:
            raise ValueError("Choose a street from the list, or Any connected street.")
        return combo.itemData(index)

    def preview_routes(self):
        self.invalidate()
        if self.network is None:
            self.load_graph()
        if self.network is None:
            return
        try:
            self.routes = self.network.routes(
                count=self.count.value() if self.action.currentData() == "add" else 1,
                seed=self.seed.value(),
                start_street=self._street(self.start_street),
                end_street=self._street(self.end_street),
                altitude_m=self.altitude.value(),
            )
            self._draw()
            lengths = [r.length_m for r in self.routes]
            self.status.setText(
                f"{len(lengths)} connected routes · {min(lengths):.0f}–{max(lengths):.0f} m · {min(lengths)/self.speed.value():.1f}–{max(lengths)/self.speed.value():.1f} s to route end. Green dots: starts; red dots: ends."
            )
            self.apply.setEnabled(
                self.action.currentData() == "selected" or self.asset.currentData() is not None
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def _draw(self):
        self.preview_map.set_context(
            streets=self.network.edges(), key=(self.network.path, self.travel_mode.currentData())
        )
        self.preview_map.set_paths(
            routes=[route.points for route in self.routes],
            proposed_current=self.routes[0].points[0] if self.routes else None,
        )
