# gpio_inputs.py
import time


class DebouncedButton:
    """
    Poll-based debounced button.
    poll() returns:
      - "pressed"  once when the debounced state becomes pressed
      - "released" once when the debounced state becomes released
      - None otherwise
    """
    def __init__(self, pin: int, *, gpio, active_high: bool = True, debounce_s: float = 0.05):
        self.pin = pin
        self.gpio = gpio
        self.active_high = active_high
        self.debounce_s = debounce_s

        # raw and stable states
        initial = self.gpio.input(self.pin)
        self._raw_last = initial
        self._stable = initial
        self._raw_changed_t = time.time()

    def _is_pressed_level(self, level: int) -> bool:
        return (level == self.gpio.HIGH) if self.active_high else (level == self.gpio.LOW)

    def poll(self):
        now = time.time()
        raw = self.gpio.input(self.pin)

        # track raw edges
        if raw != self._raw_last:
            self._raw_last = raw
            self._raw_changed_t = now

        # if raw has been stable long enough, accept it as debounced
        if (now - self._raw_changed_t) >= self.debounce_s and raw != self._stable:
            self._stable = raw
            return "pressed" if self._is_pressed_level(self._stable) else "released"

        return None


class LongPressDetector:
    """
    Debounced long-press detector.
    poll(enabled=True) returns True once per hold when held for long_press_s.
    """
    def __init__(
        self,
        pin: int,
        *,
        gpio,
        long_press_s: float = 2.0,
        active_high: bool = False,
        debounce_s: float = 0.05,
    ):
        self.pin = pin
        self.gpio = gpio
        self.long_press_s = long_press_s
        self.active_high = active_high
        self.debounce_s = debounce_s

        initial = self.gpio.input(self.pin)
        self._raw_last = initial
        self._stable = initial
        self._raw_changed_t = time.time()

        self._press_start_t = None
        self._fired = False

    def _is_pressed_level(self, level: int) -> bool:
        return (level == self.gpio.HIGH) if self.active_high else (level == self.gpio.LOW)

    def poll(self, enabled: bool = True) -> bool:
        now = time.time()
        raw = self.gpio.input(self.pin)

        if raw != self._raw_last:
            self._raw_last = raw
            self._raw_changed_t = now

        # update debounced stable state
        if (now - self._raw_changed_t) >= self.debounce_s and raw != self._stable:
            prev = self._stable
            self._stable = raw

            # debounced edge: released -> pressed
            if self._is_pressed_level(self._stable) and not self._is_pressed_level(prev):
                self._press_start_t = now
                self._fired = False

            # debounced edge: pressed -> released
            if (not self._is_pressed_level(self._stable)) and self._is_pressed_level(prev):
                self._press_start_t = None
                self._fired = False

        # if disabled, don't fire; also reset the timer so it doesn't "owe" a trigger
        if not enabled:
            self._press_start_t = None
            self._fired = False
            return False

        # held long enough
        if (
            self._press_start_t is not None
            and not self._fired
            and self._is_pressed_level(self._stable)
            and (now - self._press_start_t) >= self.long_press_s
        ):
            self._fired = True
            return True

        return False
