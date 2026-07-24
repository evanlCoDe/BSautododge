"""Small wrapper around pynput for sending global keyboard events."""

from __future__ import annotations

from typing import Union

from pynput.keyboard import Controller, Key


KeyName = Union[str, Key]


class KeyboardEvents:
    """Send keyboard input to the currently focused application.

    On macOS, grant the terminal/Python application Accessibility permission
    in System Settings before using this class to control another application.
    """

    _SPECIAL_KEYS = {
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "space": Key.space,
        "enter": Key.enter,
        "escape": Key.esc,
        "tab": Key.tab,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "alt": Key.alt,
        "cmd": Key.cmd,
    }

    def __init__(self) -> None:
        self._controller = Controller()
        self._held_keys: set[KeyName] = set()

    def press(self, key: KeyName) -> None:
        """Tap ``key`` once (for example ``"space"`` or ``"a"``)."""
        resolved_key = self._resolve_key(key)
        self._controller.press(resolved_key)
        self._controller.release(resolved_key)

    def key_down(self, key: KeyName) -> None:
        """Hold ``key`` down until :meth:`key_up` is called."""
        resolved_key = self._resolve_key(key)
        if resolved_key not in self._held_keys:
            self._controller.press(resolved_key)
            self._held_keys.add(resolved_key)

    def key_up(self, key: KeyName) -> None:
        """Release a key previously held with :meth:`key_down`."""
        resolved_key = self._resolve_key(key)
        if resolved_key in self._held_keys:
            self._controller.release(resolved_key)
            self._held_keys.remove(resolved_key)

    def release_all(self) -> None:
        """Release every key currently held by this instance."""
        for key in tuple(self._held_keys):
            self._controller.release(key)
        self._held_keys.clear()

    @classmethod
    def _resolve_key(cls, key: KeyName) -> KeyName:
        if isinstance(key, str):
            normalized = key.lower()
            return cls._SPECIAL_KEYS.get(normalized, normalized)
        return key
