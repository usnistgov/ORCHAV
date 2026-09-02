"""Qt visualizer package for ORCHAV scenarios and frame data.

Use ``python -m visualizer`` for the interactive app. That entry point runs
the GPU preflight in ``visualizer.__main__`` before handing off to
``visualizer.visualizer.main``. Scripted offscreen image rendering lives in
``visualizer.headless``; app composition, pipelines, services, panels, and
renderer backends live under ``visualizer.src``.
"""
