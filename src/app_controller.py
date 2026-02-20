"""
Application controller for the Jetson Translator.

Implements the main runtime state machine:
- Handles button press/release events
- Manages recording lifecycle
- Triggers the translation pipeline
- Supports language re-selection via long-press
- Coordinates LCD updates and GPIO cleanup
"""

import time

class AppController:
    def __init__(
        self,
        *,
        gpio,
        lcd,
        recorder,
        pipeline,
        btn_a,
        btn_b,
        lang_longpress,
        selector,
        lang_choices,
        pin_btn_a,
        pin_btn_b,
        mic_device,
        spk_device,
        rate_hz,
        poll_s,
    ):
        self.gpio = gpio
        self.lcd = lcd
        self.recorder = recorder
        self.pipeline = pipeline
        self.btn_a = btn_a
        self.btn_b = btn_b
        self.lang_longpress = lang_longpress
        self.selector = selector
        self.lang_choices = lang_choices

        self.pin_btn_a = pin_btn_a
        self.pin_btn_b = pin_btn_b
        self.mic_device = mic_device
        self.spk_device = spk_device
        self.rate_hz = rate_hz
        self.poll_s = poll_s

        # runtime state (NO GLOBALS)
        self.lang_a_to_b = None
        self.lang_b_to_a = None
        self.name_1 = None
        self.name_2 = None

    # ---- language selection ----
    def select_languages_startup(self):
        self.lang_a_to_b, self.lang_b_to_a, self.name_1, self.name_2 = self.selector.select_pair(
            self.lang_choices, initial_index=0
        )
        print(f"[SETUP] Button A: {self.name_1} ({self.lang_a_to_b[0]}) → {self.name_2} ({self.lang_a_to_b[1]})")
        print(f"[SETUP] Button B: {self.name_2} ({self.lang_b_to_a[0]}) → {self.name_1} ({self.lang_b_to_a[1]})\n")

    def reselect_languages(self):
        print("\n[LANG SEL] Long press detected → re-entering language selection.\n")
        self.lcd.write("Change langs", "Rotate + press")

        self.lang_a_to_b, self.lang_b_to_a, self.name_1, self.name_2 = self.selector.select_pair(
            self.lang_choices, initial_index=0
        )

        self.lcd.write("Ready", "Hold button")
        print("Updated directions after re-selection:\n"
              f"- Pin {self.pin_btn_a}: {self.lang_a_to_b[0]} → {self.lang_a_to_b[1]}\n"
              f"- Pin {self.pin_btn_b}: {self.lang_b_to_a[0]} → {self.lang_b_to_a[1]}\n")

    # ---- recording ----
    def start_record(self):
        if self.recorder.is_recording:
            return
        self.recorder.start()
        self.lcd.clear_scroll()
        self.lcd.write("Recording...", "")

    def stop_record(self, src_lang, tgt_lang):
        if not self.recorder.is_recording:
            return
        self.recorder.stop()
        self.pipeline.process(self.recorder.wav_path, src_lang, tgt_lang)

    # ---- UI ----
    def show_ready(self):
        self.lcd.write("Ready", "Hold button")
        print("Ready. Hold either button to RECORD; release to STOP + TRANSLATE.\n"
              f"- Pin {self.pin_btn_a}: {self.lang_a_to_b[0]} → {self.lang_a_to_b[1]}\n"
              f"- Pin {self.pin_btn_b}: {self.lang_b_to_a[0]} → {self.lang_b_to_a[1]}\n"
              f"- mic device: {self.mic_device} @ {self.rate_hz} Hz\n"
              f"- spk device: {self.spk_device}\n")

    # ---- main loop ----
    def run(self):
        # must call before run()
        if self.lang_a_to_b is None or self.lang_b_to_a is None:
            raise RuntimeError("Languages not selected. Call select_languages_startup() before run().")

        self.show_ready()

        try:
            while True:
                # Button A
                ev_a = self.btn_a.poll()
                if ev_a == "pressed":
                    self.start_record()
                elif ev_a == "released":
                    self.stop_record(*self.lang_a_to_b)

                # Button B
                ev_b = self.btn_b.poll()
                if ev_b == "pressed":
                    self.start_record()
                elif ev_b == "released":
                    self.stop_record(*self.lang_b_to_a)

                # Long-press encoder SW to reselect languages (only when NOT recording)
                if self.lang_longpress.poll(enabled=(not self.recorder.is_recording)):
                    self.reselect_languages()

                # LCD scrolling (only when not recording, handled by lcd class)
                self.lcd.tick_scroll(is_recording=self.recorder.is_recording)

                time.sleep(self.poll_s)

        except KeyboardInterrupt:
            print("\n[EXIT] Keyboard interrupt")
        finally:
            if self.recorder.is_recording:
                # stop current recording safely using current mapping
                self.stop_record(*self.lang_a_to_b)

            # GPIO cleanup belongs at the top-level main, but we can be defensive:
            try:
                self.gpio.cleanup()
            except Exception:
                pass

            print("[GPIO] cleanup done")
            self.lcd.shutdown("Shutting down", "", hold_s=5)
