"""Scene construction boundary between ORCHAV config and Sionna RT.

``SceneService`` is the only service that loads the base Sionna scene and
creates live Sionna transmitter, receiver, and target objects.  The inputs are
already parsed scenario configs, usually originating from YAML.

This service seeds the scene with initial object state.  It does not generate or
advance mobility; ``ActorStateService`` prepares the per-step actor state, and
propagation applies that state to the live Sionna objects immediately before a
ray-tracing solve.
"""

import os
from pathlib import Path
from typing import Any, TypeAlias, cast

import mitsuba as mi
import sionna
from sionna.rt import Receiver, Transmitter, load_scene

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..scenario_entities.antenna_arrays import create_planar_array
from ..sionna_integration import orientation_to_point3f, point3f
from ..target import TargetConfig, TargetManager
from .base import BaseService

SionnaScene: TypeAlias = Any


class SceneService(BaseService):
    """Build and retain the live Sionna scene objects for one pipeline run."""

    def __init__(self, simulation_config: SimulationConfig):
        super().__init__(simulation_config)
        self.scene: SionnaScene | None = None
        self.tx_list: list[Transmitter] = []
        self.rx_list: list[Receiver] = []
        self.target_managers: list[TargetManager] = []
        self.target_objects: list[Any] = []

    def build_scene(
        self,
        tx_configs: list[TransmitterConfig],
        rx_configs: list[ReceiverConfig],
        target_configs: list[TargetConfig],
    ) -> tuple[SionnaScene, list[Transmitter], list[Receiver], list[TargetManager], list[Any]]:
        """Load the scene and add all configured devices and targets.

        The returned lists contain live Sionna objects, not copies of YAML
        config.  Downstream services keep references to these objects so
        propagation can update them frame by frame.

        Returns:
            Tuple containing (scene, tx_list, rx_list, target_managers, target_objects)
        """
        self.simulation_config.validate()
        self.logger.info(f"Loading scene: {self.simulation_config.get_scene_display_name()}")

        self.scene = self._load_base_scene()

        # Frequency, bandwidth, and temperature are Sionna scene properties;
        # they must be set before downstream solvers or coverage helpers run.
        self._set_scene_frequency()

        # Apply ORCHAV scene-material defaults before generated targets exist.
        self._apply_scene_material_defaults()

        # Ensure antenna arrays
        self._ensure_antenna_arrays()

        # Device configs describe initial state from YAML.  Later timeline and
        # propagation stages are responsible for per-frame movement.
        self._add_transmitters(tx_configs)
        self._add_receivers(rx_configs)
        self._create_targets(target_configs)

        # Apply explicit material overrides after targets are added so target
        # aliases such as mat-itu_glass_Pedestrian can resolve if requested.
        self._apply_material_overrides()

        return (
            self._require_scene(),
            self.tx_list,
            self.rx_list,
            self.target_managers,
            self.target_objects,
        )

    def cleanup(self) -> None:
        """Release scene references held after a pipeline run."""
        self.scene = None
        self.tx_list = []
        self.rx_list = []
        self.target_managers = []
        self.target_objects = []

    def _require_scene(self) -> SionnaScene:
        """Return the loaded Sionna scene or fail before mutation."""
        if self.scene is None:
            raise ValueError("Scene must be loaded before it can be configured")
        return self.scene

    def _load_base_scene(self) -> SionnaScene:
        """Load the base scene from either scenario XML or Sionna's library."""
        if self.simulation_config.is_custom_xml():
            return self._load_custom_xml()
        else:
            return self._load_library_scene()

    def _load_custom_xml(self) -> SionnaScene:
        """Load a scenario-provided Mitsuba XML scene."""
        self.logger.info(f"Custom XML scene: {self.simulation_config.scene_name}")

        scene_path = Path(self.simulation_config.scene_name)
        abs_scene_path = scene_path.resolve() if scene_path.exists() else None
        cwd = os.getcwd()

        if not scene_path.exists():
            raise ValueError(f"Custom XML file not found: {self.simulation_config.scene_name}")

        # Sionna scene loading relies on a Mitsuba variant.  Some entry points
        # reach the service before a variant has been selected.
        self._ensure_mitsuba_initialized()

        try:
            scene = load_scene(self.simulation_config.scene_name)
            self.logger.info(f"Loaded custom XML scene: {self.simulation_config.scene_name}")
            return scene
        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error(f"Failed to load scene: {e}\nPath: {abs_scene_path}\nCWD: {cwd}")
            raise

    def _ensure_mitsuba_initialized(self):
        """Select the default Mitsuba variant when the process has none."""
        try:
            variant = mi.variant()
            if variant is None:
                self.logger.debug("Mitsuba3 variant not set, attempting initialization...")
                try:
                    mi.set_variant("llvm_ad_mono_polarized")
                    self.logger.debug(f"Mitsuba3 initialized with variant: {mi.variant()}")
                except (ValueError, RuntimeError) as e:
                    self.logger.warning(f"Failed to initialize Mitsuba3 variant: {e}")
        except ImportError:
            self.logger.warning("Could not import mitsuba for initialization check")

    def _load_library_scene(self) -> SionnaScene:
        """Load a scene from sionna.rt.scene library."""
        scene_name = self.simulation_config.scene_name
        try:
            scene_obj = getattr(sionna.rt.scene, scene_name)
            scene = load_scene(scene_obj)
            self.logger.info(f"Loaded library scene: {scene_name}")
            return scene
        except AttributeError as exc:
            # The advisory scene list can differ from the installed Sionna RT
            # library, so report an unavailable library scene clearly.
            self.logger.error(f"Scene '{scene_name}' not found in sionna.rt.scene")
            raise ValueError(f"Scene '{scene_name}' not available in this build") from exc
        except (OSError, RuntimeError, ValueError) as e:
            raise ValueError(f"Failed to load scene '{scene_name}': {e}") from e

    def _set_scene_frequency(self):
        """Set the carrier frequency and bandwidth for the scene from simulation config."""
        scene = self._require_scene()

        carrier_hz = self.simulation_config.carrier_frequency_hz
        mi_float = cast(Any, mi.Float)
        scene.frequency = mi_float(carrier_hz)
        scene.bandwidth = mi_float(self.simulation_config.bandwidth_hz)
        self.logger.info(
            f"Set scene carrier frequency: {carrier_hz/1e9:.3f} GHz "
            f"(bandwidth: {self.simulation_config.bandwidth_hz/1e9:.3f} GHz)"
        )

        temperature_k = getattr(self.simulation_config, "temperature_k", None)
        if temperature_k is not None:
            scene.temperature = mi_float(temperature_k)
            self.logger.info(f"Set scene temperature: {temperature_k} K")

    def _ensure_antenna_arrays(self):
        """Install default planar arrays when the loaded scene has none."""
        scene = self._require_scene()
        if not getattr(scene, "tx_array", None):
            scene.tx_array = create_planar_array(self.simulation_config.tx_antenna)
        if not getattr(scene, "rx_array", None):
            scene.rx_array = create_planar_array(self.simulation_config.rx_antenna)

    def _apply_scene_material_defaults(self):
        """Apply global material defaults before generated target aliases exist."""
        preset = getattr(
            self.simulation_config,
            "scene_material_scattering_coefficient_preset",
            "none",
        )
        if preset == "none":
            return

        try:
            from ..materials import apply_material_settings

            apply_material_settings(
                self._require_scene(),
                default_scattering_preset=preset,
                material_overrides=None,
            )
        except Exception:
            self.logger.exception("Requested scene material defaults failed")
            raise

    def _apply_material_overrides(self):
        """Apply explicit material overrides after all scene objects exist."""
        material_overrides = getattr(self.simulation_config, "material_overrides", None)
        if not material_overrides:
            return

        try:
            from ..materials import apply_material_settings

            apply_material_settings(
                self._require_scene(),
                default_scattering_preset="none",
                material_overrides=material_overrides,
            )
        except Exception:
            self.logger.exception("Requested material overrides failed")
            raise

    def _add_transmitters(self, configs: list[TransmitterConfig]):
        """Create live Sionna transmitters from parsed transmitter configs."""
        scene = self._require_scene()
        self.tx_list = []
        for config in configs:
            # YAML config carries the starting position and optional transmit
            # power.  Orientation starts at zero because per-frame orientation
            # is supplied later by prepared actor state.
            tx_kwargs: dict = dict(
                name=config.name,
                position=point3f(config.initial_position),
                orientation=orientation_to_point3f((0.0, 0.0, 0.0)),
            )
            if getattr(config, "power_dbm", None) is not None:
                tx_kwargs["power_dbm"] = config.power_dbm
            tx = Transmitter(**tx_kwargs)
            scene.add(tx)
            self.tx_list.append(tx)
            power_info = (
                f", power={config.power_dbm} dBm"
                if getattr(config, "power_dbm", None) is not None
                else ""
            )
            self.logger.info(f"Added TX '{config.name}' at {config.initial_position}{power_info}")

    def _add_receivers(self, configs: list[ReceiverConfig]):
        """Create live Sionna receivers from parsed receiver configs."""
        scene = self._require_scene()
        self.rx_list = []
        for config in configs:
            rx = Receiver(
                name=config.name,
                position=point3f(config.initial_position),
                orientation=orientation_to_point3f((0.0, 0.0, 0.0)),
            )
            scene.add(rx)
            self.rx_list.append(rx)
            self.logger.info(f"Added RX '{config.name}' at {config.initial_position}")

    def _create_targets(self, configs: list[TargetConfig]):
        """Create target managers and their live scene objects."""
        scene = self._require_scene()
        self.target_managers = []
        self.target_objects = []
        material_overrides = getattr(self.simulation_config, "material_overrides", None)
        for config in configs:
            # TargetManager is the generator-side owner for target geometry,
            # mesh updates, material aliases, and per-frame metadata.  The
            # returned object is the live Sionna scene object.
            manager = TargetManager(
                config,
                scene,
                material_overrides=material_overrides,
            )
            obj = manager.create_target()
            self.target_managers.append(manager)
            self.target_objects.append(obj)
            self.logger.info(f"Added target '{config.name}'")
