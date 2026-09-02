"""Undo-stack boundary for authoring document commands.

The domain remains usable in headless tests through :class:`CommandStack`.
The workspace can provide :class:`QtUndoStackAdapter` so the exact same
commands participate in Qt's action/shortcut and clean-index machinery.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class UndoCommand(Protocol):
    """Small command contract shared by pure-Python and Qt stacks."""

    @property
    def text(self) -> str: ...

    def redo(self) -> None: ...

    def undo(self) -> None: ...


@runtime_checkable
class UndoStack(Protocol):
    """Operations ScenarioDocument needs from an undo implementation."""

    def push(self, command: UndoCommand) -> None: ...

    def undo(self) -> None: ...

    def redo(self) -> None: ...

    def clear(self) -> None: ...

    def can_undo(self) -> bool: ...

    def can_redo(self) -> bool: ...

    def undo_text(self) -> str: ...

    def redo_text(self) -> str: ...

    def set_clean(self) -> None: ...

    def is_clean(self) -> bool: ...


class CommandStack:
    """Deterministic Qt-independent undo stack."""

    def __init__(self) -> None:
        self._commands: list[UndoCommand] = []
        self._index = 0
        self._clean_index = 0

    def push(self, command: UndoCommand) -> None:
        """Execute and append *command*, discarding any redo branch."""
        if self._index < len(self._commands):
            del self._commands[self._index :]
            if self._clean_index > self._index:
                self._clean_index = -1
        command.redo()
        self._commands.append(command)
        self._index += 1

    def undo(self) -> None:
        """Undo the latest command when available."""
        if not self.can_undo():
            return
        self._index -= 1
        self._commands[self._index].undo()

    def redo(self) -> None:
        """Redo the next command when available."""
        if not self.can_redo():
            return
        self._commands[self._index].redo()
        self._index += 1

    def clear(self) -> None:
        """Remove all undo history and mark the empty stack clean."""
        self._commands.clear()
        self._index = 0
        self._clean_index = 0

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._commands)

    def undo_text(self) -> str:
        return self._commands[self._index - 1].text if self.can_undo() else ""

    def redo_text(self) -> str:
        return self._commands[self._index].text if self.can_redo() else ""

    def set_clean(self) -> None:
        """Mark the current command index as the persisted document state."""

        self._clean_index = self._index

    def is_clean(self) -> bool:
        return self._clean_index == self._index

    @property
    def count(self) -> int:
        """Return the number of retained commands (primarily for diagnostics)."""
        return len(self._commands)


class QtUndoStackAdapter:
    """Adapt a PySide6 ``QUndoStack`` to the headless command protocol."""

    def __init__(self, stack=None) -> None:
        from PySide6.QtGui import QUndoStack

        self.stack = stack if stack is not None else QUndoStack()

    def push(self, command: UndoCommand) -> None:
        """Wrap a domain command so Qt owns its execution and action text."""

        from PySide6.QtGui import QUndoCommand

        class _WrappedCommand(QUndoCommand):
            def __init__(self) -> None:
                super().__init__(command.text)

            def redo(self) -> None:
                command.redo()

            def undo(self) -> None:
                command.undo()

        self.stack.push(_WrappedCommand())

    def undo(self) -> None:
        self.stack.undo()

    def redo(self) -> None:
        self.stack.redo()

    def clear(self) -> None:
        self.stack.clear()

    def can_undo(self) -> bool:
        return self.stack.canUndo()

    def can_redo(self) -> bool:
        return self.stack.canRedo()

    def undo_text(self) -> str:
        return self.stack.undoText()

    def redo_text(self) -> str:
        return self.stack.redoText()

    def set_clean(self) -> None:
        self.stack.setClean()

    def is_clean(self) -> bool:
        return self.stack.isClean()
