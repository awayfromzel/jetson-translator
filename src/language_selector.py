"""
Rotary encoder language selection UI.

Allows users to rotate to browse language options and
press to select source/target language pairs.
"""

import time

class LanguageSelector:
    """
    Rotary encoder language selection UI.
    - Rotate: change selection
    - Press SW (active-low): select
    """

    def __init__(
        self,
        *,
        gpio,
        lcd,
        clk_pin: int,
        dt_pin: int,
        sw_pin: int,
        debounce_s: float = 0.05,
        poll_s: float = 0.005,
        sw_active_low: bool = True,
    ):
        self.gpio = gpio
        self.lcd = lcd
        self.clk_pin = clk_pin
        self.dt_pin = dt_pin
        self.sw_pin = sw_pin
        self.debounce_s = debounce_s
        self.poll_s = poll_s
        self.sw_active_low = sw_active_low

    def _sw_pressed(self, level: int) -> bool:
        # Encoder SW is active-low in current wiring/code
        return level == self.gpio.LOW if self.sw_active_low else level == self.gpio.HIGH

    def select_one(self, prompt_line1: str, options, initial_index: int = 0):
        """
        options: list[(name, code)]
        Returns: (name, code)
        """
        self.lcd.clear_scroll()

        index = initial_index % len(options)
        name, code = options[index]
        self.lcd.write(prompt_line1, name)
        print(f"[LANG SEL] {prompt_line1} {name}")

        last_clk = self.gpio.input(self.clk_pin)

        # Ensure we start from a released state so idle LOW doesn't auto-select
        start_settle = time.time()
        while self._sw_pressed(self.gpio.input(self.sw_pin)) and (time.time() - start_settle) < 0.5:
            time.sleep(0.01)

        last_sw = self.gpio.input(self.sw_pin)

        while True:
            clk = self.gpio.input(self.clk_pin)
            dt = self.gpio.input(self.dt_pin)
            sw = self.gpio.input(self.sw_pin)

            # Rotation (detect CLK edge)
            if clk != last_clk:
                if dt != clk:
                    index = (index + 1) % len(options)
                else:
                    index = (index - 1) % len(options)
                last_clk = clk

                name, code = options[index]
                self.lcd.write(prompt_line1, name)
                print(f"[LANG SEL] -> {name}")
                time.sleep(0.01)

            # Press: edge-detect + debounce
            if sw != last_sw:
                time.sleep(self.debounce_s)
                sw2 = self.gpio.input(self.sw_pin)
                if sw2 != last_sw:
                    last_sw = sw2
                    if self._sw_pressed(sw2):
                        print(f"[LANG SEL] Selected: {name}")
                        # wait for release so we don't double-trigger
                        while self._sw_pressed(self.gpio.input(self.sw_pin)):
                            time.sleep(0.01)
                        return name, code

            time.sleep(self.poll_s)

    def select_pair(self, choices, initial_index: int = 0):
        """
        Returns:
          lang_a_to_b: (code1, code2)
          lang_b_to_a: (code2, code1)
          name1, name2
        """
        print("\n[SETUP] Use rotary encoder to choose languages.")
        print("        Rotate to change, press knob to select.\n")

        name_1, code_1 = self.select_one("Select Lang 1:", choices, initial_index=initial_index)
        name_2, code_2 = self.select_one("Select Lang 2:", choices, initial_index=initial_index)

        lang_a_to_b = (code_1, code_2)
        lang_b_to_a = (code_2, code_1)

        return lang_a_to_b, lang_b_to_a, name_1, name_2
