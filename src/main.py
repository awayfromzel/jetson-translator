#!/usr/bin/env python3
import os
import Jetson.GPIO as GPIO

from build_app import build_controller

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    AUDIO_OUT_DIR = os.path.join(PROJECT_ROOT, "audio_files")

    config = {
        "BASE_DIR": BASE_DIR,
        "PROJECT_ROOT": PROJECT_ROOT,

        "MIC_DEVICE": "plughw:CARD=Device,DEV=0",
        "SPK_DEVICE": "plughw:CARD=Audio,DEV=0",
        "PULSE_SINK": None,
        #"SPK_DEVICE": None,
        #"SPK_DEVICE": "pulse",
        #"PULSE_SINK": "ausb-KTMicro_KT_USB_Audio_2021-06-07",
        "RATE": 16000,
        "FORMAT": "S16_LE",
        "MODEL_ID": "facebook/nllb-200-distilled-600M",

        "PIN_MODE": GPIO.BOARD,
        "BTN_PIN_A": 29,
        "BTN_PIN_B": 31,
        "DEBOUNCE_S": 0.05,
        "POLL_S": 0.005,
        "ACTIVE_HIGH": True,

        "ENC_CLK_PIN": 32,
        "ENC_DT_PIN": 33,
        "ENC_SW_PIN": 7,
        "ENC_LONG_PRESS_S": 2.0,

        "LCD_COLS": 16,
        "LCD_ROWS": 2,
        "SCROLL_INTERVAL": 2,
        "LCD_I2C_ADDR": 0x27,
        "LCD_I2C_PORT": 7,
        "LCD_CHARMAP": "A00",
        "LCD_AUTO_LINEBREAKS": True,

        "PIPER_BIN": "piper",
        "VOICE_MAP": {
            "eng_Latn": "en_GB-cori-high",
            "ita_Latn": "it_IT-paola-medium",
            "spa_Latn": "es_MX-claude-high",
            "por_Latn": "pt_BR-faber-medium",
            "fra_Latn": "fr_FR-siwis-medium",
        },

        "LANG_CHOICES": [
            ("English", "eng_Latn"),
            ("Italian", "ita_Latn"),
            ("Spanish", "spa_Latn"),
            ("Portuguese", "por_Latn"),
            ("French", "fra_Latn"),
        ],

        "WARMUP": True,
    }

    controller = build_controller(GPIO=GPIO, config=config, audio_out_dir=AUDIO_OUT_DIR)
    controller.select_languages_startup()
    controller.run()

if __name__ == "__main__":
    main()
