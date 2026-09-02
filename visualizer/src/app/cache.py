"""Small app-level caches used during deferred visualizer composition."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from ..services.viewmodel_warmer import ViewModelWarmer


class ObservableCache(OrderedDict):
    """OrderedDict cache that invalidates warmed ViewModels when cleared.

    Composition still uses normal ``OrderedDict`` behavior for LRU-style cache
    operations; this subclass only adds the warmer notification needed when the
    main ViewModel cache is reset by scenario or frame transitions.
    """

    def __init__(self) -> None:
        """Create an empty cache with no attached warmer."""
        super().__init__()
        self._warmer: Optional[ViewModelWarmer] = None

    def set_warmer(self, warmer: ViewModelWarmer) -> None:
        """Attach the warmer that mirrors this cache's derived payloads."""
        self._warmer = warmer

    def clear(self) -> None:
        """Clear cached entries and invalidate warmed ViewModel payloads."""
        super().clear()
        if self._warmer is not None:
            self._warmer.invalidate(reason="cache cleared")
