"""Small presentation recipes assembled from capabilities of the loaded scene."""

PRESETS = {
    "explore": (
        "Explore",
        "Keep your current look, gently orbit the scene and loop any recorded motion.",
    ),
    "presentation": (
        "Presentation",
        "Compare scene and path looks while following the scenario tour or a gentle orbit.",
    ),
    "mpc_story": (
        "MPC story",
        "Follow slowed path pulses through several looks, then play the recording once.",
    ),
    "sensing_story": (
        "Sensing story",
        "Replay saved detections and tracks, pausing to explain a recorded detection.",
    ),
    "sensing_presentation": (
        "Sensing presentation",
        "Loop saved sensing results in Detailed appearance, with a range–velocity map and range profile beside the scene.",
    ),
    "follow_targets": (
        "Follow targets",
        "Look at each recorded target in turn while its motion plays, keeping your current look.",
    ),
    "custom": ("Custom", "Choose the camera, playback and look options under Advanced options."),
}


def available_presets(capabilities: dict[str, bool]) -> tuple[str, ...]:
    """Offer only recipes with their required content, while retaining manual setup."""
    return tuple(
        name
        for name in PRESETS
        if name
        not in {
            "presentation",
            "mpc_story",
            "sensing_story",
            "sensing_presentation",
            "follow_targets",
        }
        or capabilities.get(
            {
                "presentation": "appearance",
                "mpc_story": "showcase",
                "sensing_story": "sensing_story",
                "sensing_presentation": "sensing_presentation",
                "follow_targets": "targets",
            }[name],
            False,
        )
    )


def preset_options(name: str, capabilities: dict[str, bool]) -> dict[str, bool]:
    """Choose valid independent options; a missing tour falls back to an orbit."""
    use_tour = name in {"presentation", "mpc_story"} and capabilities.get("tour", False)
    return {
        "orbit": name not in {"follow_targets", "sensing_story", "sensing_presentation"}
        and not use_tour
        and capabilities.get("orbit", False),
        "tour": use_tour,
        "targets": name == "follow_targets" and capabilities.get("targets", False),
        "playback": capabilities.get("playback", False),
        "appearance": name == "presentation" and capabilities.get("appearance", False),
        "showcase": name == "mpc_story" and capabilities.get("showcase", False),
        "sensing_story": name == "sensing_story" and capabilities.get("sensing_story", False),
        "exhibition": name == "sensing_presentation" and capabilities.get("exhibition", False),
        "clean_view": name != "sensing_presentation" and capabilities.get("clean_view", False),
        "captions": capabilities.get("captions", False),
    }
