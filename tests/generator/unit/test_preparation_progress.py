"""Terminal and redirected progress for expensive generator startup stages."""

import io

import pytest


@pytest.mark.parametrize("terminal,interval", [(True, 0.25), (False, 5.0)])
def test_preparation_progress_throttles_but_always_reports_stage_boundaries(
    monkeypatch, terminal, interval
):
    from generator.core.pipeline import progress

    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: terminal)
    monkeypatch.setattr(progress.sys, "stderr", stream)
    now = [10.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    reporter = progress.StderrPreparationProgress()
    reporter("Verifying resources", 0, 3)
    now[0] += interval / 2
    reporter("Verifying resources", 1, 3)
    now[0] += interval
    reporter("Verifying resources", 2, 3)
    reporter("Verifying resources", 3, 3)
    reporter("Binding poses", 0, 1)
    reporter("Binding poses", 1, 1)
    reporter.close()

    output = stream.getvalue()
    assert "0/3" in output and "2/3" in output and "3/3" in output
    assert "1/3" not in output
    assert "0/1" in output and "1/1" in output
    assert "eta --:--" in output
    assert ("\r" in output) is terminal
    assert output.count("\n") == (2 if terminal else 5)
    assert output.endswith("\n")


def test_preparation_progress_closes_partial_line_without_claiming_completion(monkeypatch):
    from generator.core.pipeline import progress

    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.setattr(progress.sys, "stderr", stream)
    reporter = progress.StderrPreparationProgress()
    reporter("Validating poses", 0, 100)
    reporter.close()
    reporter.close()
    assert stream.getvalue().endswith("\n")
    assert stream.getvalue().count("\n") == 1
    assert "100/100" not in stream.getvalue()


def test_preparation_progress_tolerates_broken_display_stream(monkeypatch):
    from generator.core.pipeline import progress

    class BrokenStream(io.StringIO):
        def write(self, value):
            raise OSError("closed display")

    monkeypatch.setattr(progress.sys, "stderr", BrokenStream())
    reporter = progress.StderrPreparationProgress()
    reporter("Verifying resources", 0, 1)
    reporter("Verifying resources", 1, 1)
    reporter.close()
    progress.stderr_status("Starting")
