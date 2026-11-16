"""Shared coverage reporting for GUI and offline sensing preparation."""

from __future__ import annotations


def prepared_coverage_messages(capabilities: dict) -> tuple[str, ...]:
    """Explain every prepared spectral component with limited or failed coverage."""
    labels = {
        "saved-stft": "Saved STFT",
        "summed-rd": "Doppler over time",
        "range-profile": "Range–time",
    }
    messages = []
    for row in capabilities.get("motion", ()):
        state = row["state"]
        if state == "complete" or (state == "unavailable" and not row["available_results"]):
            continue
        name = labels.get(row["family"], row["family"])
        message = (
            f"{name} ({row['variant']}): "
            f"{row['loaded_results']:,}/{row['available_results']:,} saved results"
        )
        span = row.get("loaded_time_span")
        if span is not None:
            message += f" · {span[0]:.6g}–{span[1]:.6g} s"
        reason = row.get("reason")
        if reason:
            message += f" · {reason}"
        if state == "limited":
            message += ". Increase --sensing-overview-cache-mb and reopen to extend coverage."
        messages.append(message)
    return tuple(messages)
