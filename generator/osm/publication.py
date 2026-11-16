"""Stage a complete build before replacing generated assets; roll back errors."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_scene(output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".osm-build-", dir=root) as temporary:
        stage = Path(temporary) / "new"
        backup = Path(temporary) / "previous"
        stage.mkdir()
        yield stage
        # Preserve unrelated files, including targets stored under mesh/.
        # Only generated files are replaced; metadata is published last.
        files = sorted(
            (p for p in stage.rglob("*") if p.is_file()),
            key=lambda p: ({"scene.xml": 1, "osm_metadata.json": 2}.get(p.name, 0), str(p)),
        )
        replaced = []
        try:
            for source in files:
                relative = source.relative_to(stage)
                target = root / relative
                previous = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                existed = target.exists()
                if existed:
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, previous)
                os.replace(source, target)
                replaced.append((target, previous, existed))
        except BaseException:
            for target, previous, existed in reversed(replaced):
                if existed:
                    os.replace(previous, target)
                else:
                    target.unlink(missing_ok=True)
            raise
