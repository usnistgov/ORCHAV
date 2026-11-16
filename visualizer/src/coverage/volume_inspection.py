"""Coverage source-cell descriptions and unsaturated ray-maximum sampling."""

import numpy as np
from matplotlib import colormaps

from shared.coverage.availability import CoverageAvailability
from shared.coverage.volume import ALL_TRANSMITTERS

from ..types.render_payloads import VolumePayload
from .analysis import (
    coverage_metric_color_scale,
    coverage_metric_comparator,
    coverage_metric_label,
    coverage_metric_valid_mask,
    serving_tx_color_rgb,
)


def scalar_template(
    model,
    low,
    high,
    *,
    colormap="viridis",
    show_nonfinite=False,
    visible_mask=None,
    solid_colour=None,
):
    """Keep finite source dB samples exact; clip colour, never MIP ordering.

    Two lower sentinel bins distinguish missing samples from saved zero solver
    contributions. They are display-only; inspection reads the immutable model.
    """
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Colour minimum must be below maximum")
    # The renderer projects maxima. Negating loss selects the lowest loss along
    # a ray while inspection and the colour legend retain physical loss units.
    quantity = getattr(model, "quantity", "path_gain_db")
    loss = coverage_metric_comparator(quantity) == "<="
    logarithmic = coverage_metric_color_scale(quantity) == "logarithmic"
    valid = coverage_metric_valid_mask(model.values, quantity)
    zeros = getattr(model, "zero_contributions", np.isinf(model.values))
    valid &= ~zeros
    strength = -model.values if loss else model.values
    if logarithmic:
        if low <= 0:
            raise ValueError("Logarithmic colour limits must be positive")
        with np.errstate(divide="ignore", invalid="ignore"):
            strength = np.log10(strength)
        low, high = np.log10([low, high])
    strength = np.where(valid, strength, np.where(zeros, -np.inf, np.nan))
    finite = strength[np.isfinite(strength)]
    minimum = float(finite.min()) if finite.size else low
    maximum = float(finite.max()) if finite.size else high
    span = max(maximum - minimum, 1.0)
    sentinels = 3 if visible_mask is not None else 2
    step = span / (1023 - sentinels)
    absent, zero = minimum - 2 * step, minimum - step
    bottom = minimum - sentinels * step
    top = minimum + span
    values = np.where(np.isnan(strength), absent, np.where(np.isneginf(strength), zero, strength))
    if visible_mask is not None:
        # Filtered/out-of-region cells stay transparent even when missing
        # source cells are deliberately grey. Never conflate those meanings.
        values = np.where(visible_mask, values, bottom)
    # Nearest palette centres reproduce the same locked scientific colour scale.
    samples = np.linspace(bottom, top, 1024)
    if loss:
        samples = -samples
    index = np.clip(np.floor((samples - low) / (high - low) * 255), 0, 255).astype(int)
    lut = colormaps[colormap](np.linspace(0, 1, 256))
    palette = lut[index].copy()
    if solid_colour is not None:
        palette[:] = solid_colour
    # A neutral swatch sits outside the scientific scale; inspection retains
    # the distinct zero-contribution and unavailable source identities.
    palette[sentinels - 2 : sentinels] = (0.5, 0.5, 0.5, 1.0) if show_nonfinite else 0
    if visible_mask is not None:
        palette[0] = 0
    return VolumePayload(values, palette, (bottom, top), threshold=minimum - step / 2)


def categorical_template(model, *, show_nonfinite=False, transmitter_index=None):
    """Keep serving identities discrete; zero is hidden, one denotes no server.

    A selected transmitter produces its region. Other valid transmitter IDs
    remain hidden, not relabelled as unavailable. Projection by maximum ID is
    deliberately unsupported by the coverage controls.
    """
    count = len(model.transmitter_names)
    raw = model.values
    valid = np.isfinite(raw) & (raw >= 0) & (raw < count) & (raw == np.floor(raw))
    values = np.where(valid, raw + 2, 1 if show_nonfinite else 0)
    if transmitter_index is not None:
        values = np.where(valid & (raw != transmitter_index), 0, values)
    palette = np.zeros((count + 2, 4), dtype=np.float32)
    palette[1] = (0.5, 0.5, 0.5, 1) if show_nonfinite else 0
    for index in range(count):
        palette[index + 2, :3] = serving_tx_color_rgb(index)
        palette[index + 2, 3] = 1
    return VolumePayload(values, palette, (0, count + 1))


def describe_cell(model, indices, origin):
    """Format only source facts; coverage cells have no sensing trial outcome."""
    x, y, z = indices
    raw = model.values[z, y, x]
    _, unit = coverage_metric_label(model.quantity)
    value = "unavailable (NaN)" if np.isnan(raw) else f"{raw:g} {unit}"
    if model.quantity == "serving_tx":
        value = (
            model.transmitter_names[int(raw)]
            if np.isfinite(raw) and 0 <= raw < len(model.transmitter_names) and raw == int(raw)
            else "no serving transmitter"
        )
    reason = "unknown (not recorded)"
    if model.availability is not None:
        reason = {
            CoverageAvailability.UNKNOWN: "unknown; historical reason not recorded",
            CoverageAvailability.POSITIVE: "positive saved estimate",
            CoverageAvailability.ZERO_CONTRIBUTION: "zero solver contribution; not proof of no physical signal",
            CoverageAvailability.INVALID: "invalid saved estimate",
        }.get(int(model.availability[z, y, x]), "unknown (not recorded)")
    tx = model.transmitter
    if tx == ALL_TRANSMITTERS:
        tx = (
            "all transmitters; aggregate defined by the selected metric"
            if getattr(model, "transmitter_names", ())
            else "all transmitters, strongest-link aggregate; winning TX identity not retained"
        )
    centre = ", ".join(
        f"{axis} {values[index]:g}"
        for axis, values, index in zip("XYZ", model.axes, indices, strict=True)
    )
    return (
        f"{origin}\nSource index XYZ: {x}, {y}, {z}\nCentre: {centre} m\n"
        f"Raw {model.quantity}: {value}\nTransmitter: {tx}\nAvailability: {reason}\n"
        "Saved radio-map cell estimate; no sensing detection outcome or trial."
    )


def cell_summary(model, indices):
    """Two-line readout; source indices and provenance stay in the tooltip."""
    value = model.values[tuple(reversed(indices))]
    label, unit = coverage_metric_label(model.quantity)
    if model.quantity == "serving_tx":
        text = (
            model.transmitter_names[int(value)]
            if np.isfinite(value)
            and 0 <= value < len(model.transmitter_names)
            and value == int(value)
            else "None"
        )
    elif np.isneginf(value) or model.zero_contributions[tuple(reversed(indices))]:
        text = "Zero contribution"
    elif not np.isfinite(value):
        text = "Unavailable"
    else:
        text = f"{value:.4g} {unit}".strip()
    location = " · ".join(
        f"{name} {axis[index]:g}"
        for name, axis, index in zip("XYZ", model.axes, indices, strict=True)
    )
    return f"{location} m\n{label}: {text}"
