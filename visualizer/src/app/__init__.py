"""Application composition layer for the Qt visualizer.

The root ``OrchavVisualizer`` class delegates startup and scenario work here:
``startup_workflow`` owns CLI launch, ``state_bootstrap`` installs runtime
attributes, ``services`` builds the service/controller graph, ``composition``
finishes deferred UI setup, and ``scenario_workflow`` coordinates scenario
cleanup and loading. Renderer-specific behavior stays behind the renderer
protocol and backend packages.
"""
