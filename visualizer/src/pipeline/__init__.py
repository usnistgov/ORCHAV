"""Frame pipeline and ViewModel construction internals.

``FramePipeline`` is the runtime path from a frame source to renderer payloads.
``MPCCore`` turns raw frame dictionaries into canonical MPC arrays and then a
renderer-neutral ``ViewModel``. Cache key helpers are shared with
``ViewModelWarmer`` so foreground rendering and background pre-warming agree on
which derived payload is valid.
"""
