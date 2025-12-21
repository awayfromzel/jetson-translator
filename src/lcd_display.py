# lcd_display.py
import time

try:
    from RPLCD.i2c import CharLCD
except ImportError:
    CharLCD = None


class LCDDisplay:
    def __init__(
        self,
        cols=16,
        rows=2,
        i2c_address=0x27,
        i2c_port=7,
        charmap="A00",
        auto_linebreaks=True,
        scroll_interval_s=2.0,
    ):
        self.cols = cols
        self.rows = rows
        self.i2c_address = i2c_address
        self.i2c_port = i2c_port
        self.charmap = charmap
        self.auto_linebreaks = auto_linebreaks
        self.scroll_interval_s = scroll_interval_s

        self.lcd = None

        # scrolling state
        self._scroll_text = ""
        self._scroll_offset = 0
        self._scroll_last_time = 0.0

    @property
    def is_available(self) -> bool:
        return self.lcd is not None

    def init(self):
        if CharLCD is None:
            print("[LCD] RPLCD not installed; LCD output disabled.")
            self.lcd = None
            return

        try:
            self.lcd = CharLCD(
                i2c_expander="PCF8574",
                address=self.i2c_address,
                port=self.i2c_port,
                cols=self.cols,
                rows=self.rows,
                charmap=self.charmap,
                auto_linebreaks=self.auto_linebreaks,
            )
            print(f"[LCD] Initialized at 0x{self.i2c_address:02x} on bus {self.i2c_port}.")
        except Exception as e:
            print(f"[LCD] init failed: {e}")
            self.lcd = None

    def write(self, line1: str = "", line2: str = ""):
        if self.lcd is None:
            return
        try:
            self.lcd.clear()
            if line1:
                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(line1[: self.cols])
            if self.rows > 1 and line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[: self.cols])
        except Exception as e:
            print(f"[LCD] write error: {e}")

    def set_scroll_text(self, text: str):
        self._scroll_text = text or ""
        self._scroll_offset = 0
        self._scroll_last_time = time.time()

    def clear_scroll(self):
        self.set_scroll_text("")

    def tick_scroll(self, is_recording: bool):
        """
        Call this periodically. It will scroll line 2 when:
          - LCD exists
          - scroll text is non-empty
          - not currently recording
        """
        if self.lcd is None:
            return
        if not self._scroll_text:
            return
        if is_recording:
            return

        now = time.time()
        if now - self._scroll_last_time < self.scroll_interval_s:
            return
        self._scroll_last_time = now

        padded = self._scroll_text + "   "
        if self._scroll_offset >= len(padded):
            self._scroll_offset = 0

        window = padded[self._scroll_offset : self._scroll_offset + self.cols]
        if len(window) < self.cols:
            window = window.ljust(self.cols)
        self._scroll_offset += 1

        try:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(" " * self.cols)
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(window)
        except Exception as e:
            print(f"[LCD] scroll error: {e}")

    def shutdown(self, line1="Shutting down", line2="", hold_s=5):
        if self.lcd is None:
            return
        try:
            self.write(line1, line2)
            time.sleep(hold_s)
            self.lcd.clear()
        except Exception:
            pass
