"""Human-readable campaign progress, including time spent between frame callbacks."""

from __future__ import annotations

import shutil
import sys
import threading
import time

from generator.core.pipeline.progress import format_duration


class VolumeProgress:
    """Show one live terminal line, or periodic plain lines in redirected logs.

    Counts cover the selected worker partition and limit, including saved trials.
    ETA uses only trials completed during this run, so resuming cannot inflate
    the throughput. A heartbeat keeps initialization and finalization visible.
    """

    def __init__(self, label, total, *, completed=0, stream=None):
        self.label = label
        self.total = total
        self.completed = completed
        self.initial_completed = completed
        self.stream = stream if stream is not None else sys.stderr
        self.terminal = self.stream.isatty()
        self.started = time.monotonic()
        self.last_render = float("-inf")
        self.line_width = 0
        self.stage = "Starting"
        self.trial_id = ""
        self.frames = 0
        self.total_frames = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.render(force=True)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        if exc_type is not None:
            self.set_stage("Interrupted" if exc_type is KeyboardInterrupt else "Failed")
            self.message("  Completed trials are preserved; see the error below.")
        else:
            self.set_stage("Finished")
            self.message(
                f"  {self.label}: {self.completed:,}/{self.total:,} trials complete "
                f"({self.completed - self.initial_completed:,} new, "
                f"{self.initial_completed:,} already saved) in "
                f"{format_duration(time.monotonic() - self.started)}."
            )

    def _heartbeat(self):
        while not self.stop.wait(1.0):
            self.render()

    def start_trial(self, trial, total_frames):
        with self.lock:
            self.trial_id = trial["id"]
            self.frames = 0
            self.total_frames = total_frames
            self.stage = "Loading scene / preparing simulation"
        position = trial.get("position_m")
        location = f" at ({', '.join(f'{v:g}' for v in position)}) m" if position else ""
        self.message(f"  Trial {self.completed + 1:,}/{self.total:,}: {self.trial_id}{location}")
        self.render(force=True)

    def step_completed(self, info):
        with self.lock:
            self.frames += 1
            self.total_frames = info.total_steps
            self.stage = (
                "Saving frames / generating summaries"
                if self.frames == self.total_frames
                else "Generating frames"
            )
        self.render(force=self.frames in (1, self.total_frames))

    def set_stage(self, stage):
        with self.lock:
            self.stage = stage
        self.render(force=True)

    def complete_trial(self, outcome):
        with self.lock:
            self.completed += 1
            self.stage = "Trial saved"
        result = "target detected" if outcome["hit"] else "target missed"
        self.message(
            f"  Saved {self.trial_id}: {result}; {outcome['fp']} extra report(s) "
            f"| trial time {format_duration(outcome['wall_seconds'])}"
        )
        self.render(force=True)

    def update_workers(self, completed, active):
        with self.lock:
            self.completed = completed
            self.stage = f"{active} worker(s) running"
        self.render()

    def render(self, *, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_render < (0.25 if self.terminal else 15.0):
                return
            self.last_render = now
            elapsed = now - self.started
            acquired = self.completed - self.initial_completed
            remaining = self.total - self.completed
            eta = (
                format_duration(elapsed * remaining / acquired)
                if acquired > 0
                else ("00:00" if remaining == 0 else "estimating")
            )
            fraction = self.completed / self.total if self.total else 1.0
            bar = "#" * int(16 * fraction) + "-" * (16 - int(16 * fraction))
            frames = f" | frames {self.frames}/{self.total_frames}" if self.total_frames else ""
            line = (
                f"{self.label} [{bar}] {self.completed:,}/{self.total:,} "
                f"({fraction:.0%}){frames} | {self.stage} "
                f"| elapsed {format_duration(elapsed)} | ETA {eta}"
            )
            if self.terminal:
                # Avoid wrapping a carriage-return line onto the next row.
                width = shutil.get_terminal_size((120, 24)).columns - 1
                if len(line) > width:
                    line = (
                        f"{self.label} {self.completed:,}/{self.total:,} ({fraction:.0%})"
                        f"{frames} | {format_duration(elapsed)} | ETA {eta} | {self.stage}"
                    )
                line = line[: max(1, width)]
                self._write("\r" + line + " " * max(0, self.line_width - len(line)))
                self.line_width = len(line)
            else:
                self._write(line + "\n")

    def message(self, message):
        with self.lock:
            if self.line_width:
                self._write("\r" + " " * self.line_width + "\r")
                self.line_width = 0
            self._write(message + "\n")

    def _write(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
        except OSError:
            # A lost progress stream must not interrupt an acquisition.
            pass
