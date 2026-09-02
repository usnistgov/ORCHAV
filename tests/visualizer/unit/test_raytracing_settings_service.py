from __future__ import annotations

from types import SimpleNamespace

from visualizer.src.panels.data_source.raytracing_section import RaytracingControlSection
from visualizer.src.services.raytracing_settings_service import RaytracingSettingsService


def test_default_low_release_and_scaled_drag_settings() -> None:
    service = RaytracingSettingsService()

    release = service.release_settings()
    drag = service.drag_settings()

    assert release["max_depth"] == 3
    assert release["samples_per_src"] == 1_000_000
    assert release["max_num_paths_per_src"] == 500_000
    assert release["diffuse_reflection"] is False
    assert drag["max_depth"] == 2
    assert drag["samples_per_src"] == 4096
    assert drag["max_num_paths_per_src"] == 30000
    assert drag["diffuse_reflection"] is False
    assert drag["seed"] == release["seed"]


def test_custom_drag_preserves_phenomena_and_never_exceeds_release_budget() -> None:
    service = RaytracingSettingsService()
    service.set_custom(
        {
            "max_depth": 1,
            "samples_per_src": 2000,
            "max_num_paths_per_src": 9000,
            "seed": 99,
            "los": False,
            "specular_reflection": False,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "synthetic_array": True,
        }
    )

    release = service.release_settings()
    drag = service.drag_settings()

    assert drag["max_depth"] == release["max_depth"] == 1
    assert drag["samples_per_src"] <= release["samples_per_src"]
    assert drag["max_num_paths_per_src"] <= release["max_num_paths_per_src"]
    assert drag["seed"] == 99
    assert drag["los"] is False
    assert drag["specular_reflection"] is False
    assert drag["diffuse_reflection"] is True
    assert drag["refraction"] is True
    assert drag["diffraction"] is True


def test_raytracing_section_syncs_custom_widget_values_to_settings_service(qapp) -> None:
    service = RaytracingSettingsService()
    parent = SimpleNamespace(raytracing_settings_service=service)
    section = RaytracingControlSection(parent, {}, lambda: "")
    widget = section.create_content()

    section.widgets["rt_preset_combo"].setCurrentText("custom")
    section.widgets["rt_max_depth"].setValue(6)
    section.widgets["rt_samples_per_src"].setValue(150000)
    section.widgets["rt_max_num_paths_per_src"].setValue(75000)
    section.widgets["rt_seed"].setValue(321)
    section.widgets["rt_diffuse"].setChecked(False)
    section.widgets["rt_refraction"].setChecked(True)

    preset, config = section.current_raytracing_config()

    assert preset == "custom"
    assert service.current_preset == "custom"
    assert config == service.release_settings()
    assert config["max_depth"] == 6
    assert config["samples_per_src"] == 150000
    assert config["max_num_paths_per_src"] == 75000
    assert config["seed"] == 321
    assert config["diffuse_reflection"] is False
    assert config["refraction"] is True
    widget.deleteLater()


def test_raytracing_section_syncs_named_preset_to_settings_service(qapp) -> None:
    service = RaytracingSettingsService()
    parent = SimpleNamespace(raytracing_settings_service=service)
    section = RaytracingControlSection(parent, {}, lambda: "")
    widget = section.create_content()

    assert section.widgets["rt_preset_combo"].currentText() == "low"
    assert service.current_preset == "low"

    section.widgets["rt_preset_combo"].setCurrentText("low")
    preset, config = section.current_raytracing_config()

    assert preset == "low"
    assert service.current_preset == "low"
    assert config == service.release_settings()
    assert config["max_depth"] == 3
    assert config["diffuse_reflection"] is False
    widget.deleteLater()


def test_raytracing_section_loads_authored_named_preset(qapp) -> None:
    service = RaytracingSettingsService()
    parent = SimpleNamespace(raytracing_settings_service=service)
    section = RaytracingControlSection(parent, {}, lambda: "")
    widget = section.create_content()

    section.sync_from_scenario(
        SimpleNamespace(raytracing={"quality": {"preset": "low", "custom": {}}})
    )

    assert section.widgets["rt_preset_combo"].currentText() == "low"
    assert service.current_preset == "low"
    assert service.release_settings()["max_depth"] == 3
    widget.deleteLater()


def test_raytracing_section_merges_authored_custom_overrides(qapp) -> None:
    service = RaytracingSettingsService()
    parent = SimpleNamespace(raytracing_settings_service=service)
    section = RaytracingControlSection(parent, {}, lambda: "")
    widget = section.create_content()

    section.sync_from_scenario(
        SimpleNamespace(
            raytracing={
                "quality": {
                    "preset": "low",
                    "custom": {"max_depth": 6, "diffuse_reflection": True},
                }
            }
        )
    )

    assert section.widgets["rt_preset_combo"].currentText() == "custom"
    assert service.current_preset == "custom"
    settings = service.release_settings()
    assert settings["max_depth"] == 6
    assert settings["samples_per_src"] == 1_000_000
    assert settings["diffuse_reflection"] is True
    widget.deleteLater()
