"""Initialize the ordinary scene radios for a frame-free volume overview."""

import numpy as np


def initialize_scene_radios(visualizer):
    """Use authored static poses with the main node/label service.

    Volume overviews have no top-level frames to seed the normal radio markers.
    Their frozen scene actors still provide positions, names and label styles.
    Frame-backed scenes and moving actors retain their normal pose ownership.
    """
    if not getattr(visualizer, "_scene_only_mode", False):
        return
    actors = visualizer.scenario_config.actors
    groups = (actors.tx, actors.rx)
    if not all(groups) or any(
        actor.mobility.type != "stationary" for group in groups for actor in group
    ):
        return
    positions = [
        np.asarray([actor.mobility.position_m for actor in group], dtype=float) for group in groups
    ]
    visualizer.num_tx, visualizer.num_rx = map(len, groups)
    service = visualizer.node_service
    service.create_tx_rx_markers()
    service.update_available_tx_rx_from_frame(visualizer.num_tx, visualizer.num_rx)
    service.update_tx_rx_positions(*positions)
    # The overview now has real geometry. It never loads an MPC frame, so the
    # normal first-frame cleanup cannot retire the empty-scene helper sphere.
    scene = getattr(visualizer, "scene_service", None)
    if scene is not None:
        scene.remove_empty_scene_anchor()
    # These actors are inspectable without MPC frames. Keep their shared label
    # controls available even though other frame-dependent workflows stay hidden.
    manager = getattr(visualizer, "ui_manager", None)
    if manager is not None:
        manager.set_panel_visible("nodes", True)
