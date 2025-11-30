#!/usr/bin/env python3
import Jetson.GPIO as GPIO
import subprocess, signal, time, datetime, os, sys, torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shutil

# Try to import LCD library (optional)
try:
    from RPLCD.i2c import CharLCD
except ImportError:
    CharLCD = None

# ---------- CONFIG ----------
# Stable ALSA device names (use card *names*, not numbers)
MIC_DEVICE = "plughw:CARD=Device,DEV=0"   # from `arecord -l` → card 0: Device [...]
SPK_DEVICE = "plughw:CARD=Audio,DEV=0"    # from `aplay -l`   → card 1: Audio [...]
RATE = 16000 
FORMAT = "S16_LE"
MODEL_ID = "facebook/nllb-200-distilled-600M"

PIN_MODE = GPIO.BOARD
BTN_PIN_A = 29               # Button A → Direction 1
BTN_PIN_B = 31               # Button B → Direction 2
DEBOUNCE_S = 0.05
POLL_S = 0.005
ACTIVE_HIGH = True

# Rotary encoder pins (HW-040)
ENC_CLK_PIN = 32             # CLK
ENC_DT_PIN  = 33             # DT
ENC_SW_PIN  = 7              # push button (SW)

# Long-press duration (seconds) for re-entering language selection  # <<< NEW
ENC_LONG_PRESS_S = 2.0       # change this if you want longer/shorter hold  # <<< NEW

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # go up one level from src/

AUDIO_OUT_DIR = os.path.join(PROJECT_ROOT, "audio_files")
PIPER_VOICE_DIR = os.path.join(PROJECT_ROOT, "piper_voices")
PIPER_BIN = "piper"
# ---------- VOICE SELECTION ----------
VOICE_MAP = {
    "eng_Latn": "en_GB-cori-high",
    "ita_Latn": "it_IT-paola-medium",
    "spa_Latn": "es_MX-claude-high",
    "por_Latn": "pt_BR-faber-medium",
    "fra_Latn": "fr_FR-siwis-medium",
}

# ---------- TRANSLATION LANGUAGE SETTINGS ----------
LANG_A_TO_B = ("eng_Latn", "ita_Latn")   # English → Italian (overwritten after selection)
LANG_B_TO_A = ("ita_Latn", "eng_Latn")   # Italian → English

LANG_CHOICES = [
    ("English",     "eng_Latn"),
    ("Italian",     "ita_Latn"),
    ("Spanish",     "spa_Latn"),
    ("Portuguese",  "por_Latn"),
    ("French",      "fra_Latn")
]

# ---------- LCD SUPPORT ----------
LCD = None
LCD_COLS = 16
LCD_ROWS = 2

SCROLL_TEXT = "                              "
SCROLL_OFFSET = 0
SCROLL_LAST_TIME = 0.0
SCROLL_INTERVAL = 2  # seconds between scroll steps

def init_lcd():
    global LCD
    if CharLCD is None:
        print("[LCD] RPLCD not installed; LCD output disabled.")
        return
    try:
        LCD = CharLCD(
            i2c_expander='PCF8574',
            address=0x27,
            port=7,
            cols=LCD_COLS,
            rows=LCD_ROWS,
            charmap='A00',
            auto_linebreaks=True
        )
        print("[LCD] Initialized at 0x27 on bus 7.")
    except Exception as e:
        print(f"[LCD] init failed: {e}")
        LCD = None

def lcd_write_lines(line1="", line2=""):
    if LCD is None:
        return
    try:
        LCD.clear()
        if line1:
            LCD.cursor_pos = (0, 0)
            LCD.write_string(line1[:LCD_COLS])
        if LCD_ROWS > 1 and line2:
            LCD.cursor_pos = (1, 0)
            LCD.write_string(line2[:LCD_COLS])
    except Exception as e:
        print(f"[LCD] write error: {e}")

def lcd_scroll_step():
    global SCROLL_OFFSET, SCROLL_LAST_TIME

    if LCD is None:
        return
    if not SCROLL_TEXT:
        return
    if rec_proc is not None:
        return

    now = time.time()
    if now - SCROLL_LAST_TIME < SCROLL_INTERVAL:
        return
    SCROLL_LAST_TIME = now

    padded = SCROLL_TEXT + "   "
    if SCROLL_OFFSET >= len(padded):
        SCROLL_OFFSET = 0

    window = padded[SCROLL_OFFSET:SCROLL_OFFSET + LCD_COLS]
    if len(window) < LCD_COLS:
        window = window.ljust(LCD_COLS)
    SCROLL_OFFSET += 1

    try:
        LCD.cursor_pos = (1, 0)
        LCD.write_string(" " * LCD_COLS)
        LCD.cursor_pos = (1, 0)
        LCD.write_string(window)
    except Exception as e:
        print(f"[LCD] scroll error: {e}")

# ---------- INIT MODELS ----------
print("Loading ASR (faster-whisper)…")
init_lcd()
lcd_write_lines("Loading...", "ASR model")
asr = None
for dev, ctype in [
    ("cuda", "int8_float16"),
    ("cuda", "float16"),
    ("cpu",  "int8"),
]:
    try:
        asr = WhisperModel("small", device=dev, compute_type=ctype, device_index=0)
        print(f"[ASR] using {dev}/{ctype}")
        break
    except Exception as e:
        print(f"[ASR] {dev}/{ctype} failed: {e}")
if asr is None:
    raise RuntimeError("Failed to initialize faster-whisper in all modes.")

lcd_write_lines("Loading...", "MT model")
print("Loading MT (NLLB distilled)…")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mt = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None
)
if torch.cuda.is_available():
    mt = mt.to("cuda")

def lang_id(tok, code):
    if hasattr(tok, "lang_code_to_id") and isinstance(tok.lang_code_to_id, dict) and code in tok.lang_code_to_id:
        return tok.lang_code_to_id[code]
    tid = tok.convert_tokens_to_ids(code)
    if tid is not None and tid != tok.unk_token_id:
        return tid
    if hasattr(tok, "lang_code_to_token"):
        return tok.convert_tokens_to_ids(tok.lang_code_to_token[code])
    raise ValueError(f"Unknown language code: {code}")

def translate(text, src, tgt, max_new=200):
    inputs = tok(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    gen = mt.generate(**inputs, max_new_tokens=max_new, forced_bos_token_id=lang_id(tok, tgt))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def transcribe(path):
    segs, info = asr.transcribe(path, beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()

# ---------- PIPER TTS ----------
def tts_voice_paths(lang_code):
    name = VOICE_MAP.get(lang_code)
    if not name:
        return None
    onnx = os.path.join(PIPER_VOICE_DIR, f"{name}.onnx")
    cfg  = os.path.join(PIPER_VOICE_DIR, f"{name}.onnx.json")
    if not (os.path.isfile(onnx) and os.path.isfile(cfg)):
        return None
    return onnx, cfg

def speak_with_piper(text, lang_code, out_basename_prefix="tts"):
    paths = tts_voice_paths(lang_code)
    if not paths:
        print(f"[TTS] No Piper voice configured or files missing for {lang_code}. Skipping speech.")
        return
    onnx, cfg = paths
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(AUDIO_OUT_DIR, f"{out_basename_prefix}_{lang_code}_{stamp}.wav")

    try:
        print(f"[TTS] synthesizing → {wav_path} (voice: {os.path.basename(onnx)})")
        proc = subprocess.Popen(
            [PIPER_BIN, "-m", onnx, "-c", cfg, "--output-file", wav_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(text, timeout=60)
        if proc.returncode != 0:
            print(f"[TTS] Piper error (code {proc.returncode}): {stderr.strip()}")
            return
        play_cmd = ["aplay", "-D", SPK_DEVICE, wav_path]
        if shutil.which("pasuspender") is not None:
            play_cmd = ["pasuspender", "--"] + play_cmd
        subprocess.run(play_cmd, check=False)


    except Exception as e:
        print(f"[TTS] Exception running Piper: {e}")

# ---------- GPIO RECORDING ----------
rec_proc = None
press_t0 = None
current_wav = os.path.join(AUDIO_OUT_DIR, "speech.wav")

def start_record():
    global rec_proc, press_t0, SCROLL_TEXT
    if rec_proc is not None:
        return
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-f", FORMAT,
        "-c", "1",
        "-r", str(RATE),
        "-t", "wav",
        current_wav
    ]
    press_t0 = time.time()
    print(f"[REC] start → {current_wav}")
    rec_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    SCROLL_TEXT = ""
    lcd_write_lines("Recording...", "")

def stop_record(src_lang, tgt_lang):
    global rec_proc, press_t0, SCROLL_TEXT, SCROLL_OFFSET, SCROLL_LAST_TIME
    if rec_proc is None:
        return
    print("[REC] stopping…")
    rec_proc.send_signal(signal.SIGINT)
    try:
        rec_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        rec_proc.kill()
        rec_proc.wait()
    rec_proc = None

    dur = (time.time() - press_t0) if press_t0 else None
    print(f"[REC] done ({dur:.2f}s)")

    print("Transcribing…")
    lcd_write_lines("Transcribing...", "")

    text = transcribe(current_wav)
    print("[ASR]", text or "(no speech)")
    if text:
        print(f"Translating {src_lang} → {tgt_lang} …")
        out = translate(text, src_lang, tgt_lang)
        print("[MT ]", out)

        dir_label = f"{src_lang[:3]}→{tgt_lang[:3]}"
        SCROLL_TEXT = out
        SCROLL_OFFSET = 0
        SCROLL_LAST_TIME = time.time()
        lcd_write_lines(dir_label, SCROLL_TEXT[:LCD_COLS])
        speak_with_piper(out, tgt_lang, out_basename_prefix="translation")
    else:
        print("[WARN] No speech detected.")
        SCROLL_TEXT = ""
        lcd_write_lines("No speech", "Try again")

def is_pressed(level: int) -> bool:
    return (level == GPIO.HIGH) if ACTIVE_HIGH else (level == GPIO.LOW)

# ---------- ROTARY LANGUAGE SELECTION ----------
def choose_language_with_encoder(prompt_line1, options, initial_index=0):
    """
    Use the rotary encoder to pick one of `options` (list of (name, code)).
    Rotate to change, press SW to select.
    Returns (name, code).
    """
    global SCROLL_TEXT

    if LCD is not None:
        SCROLL_TEXT = ""
    index = initial_index % len(options)
    name, code = options[index]

    lcd_write_lines(prompt_line1, name)
    print(f"[LANG SEL] {prompt_line1} {name}")

    last_clk = GPIO.input(ENC_CLK_PIN)

    # --- ensure we start from a released state so idle LOW doesn't auto-select ---
    start_settle = time.time()
    while GPIO.input(ENC_SW_PIN) == GPIO.LOW and (time.time() - start_settle) < 0.5:
        time.sleep(0.01)
    last_sw = GPIO.input(ENC_SW_PIN)
    # ------------------------------------------------------------------

    while True:
        clk = GPIO.input(ENC_CLK_PIN)
        dt  = GPIO.input(ENC_DT_PIN)
        sw  = GPIO.input(ENC_SW_PIN)

        # Rotation
        if clk != last_clk:
            if dt != clk:
                index = (index + 1) % len(options)
            else:
                index = (index - 1) % len(options)
            last_clk = clk
            name, code = options[index]
            lcd_write_lines(prompt_line1, name)
            print(f"[LANG SEL] -> {name}")
            time.sleep(0.01)

        # Button press: look for an EDGE, not just level
        if sw != last_sw:
            # Debounce
            time.sleep(DEBOUNCE_S)
            sw2 = GPIO.input(ENC_SW_PIN)
            if sw2 != last_sw:
                last_sw = sw2
                if sw2 == GPIO.LOW:   # active low: this is the press
                    print(f"[LANG SEL] Selected: {name}")
                    # wait for release so we don't double-trigger
                    while GPIO.input(ENC_SW_PIN) == GPIO.LOW:
                        time.sleep(0.01)
                    return name, code

        time.sleep(POLL_S)

def setup_language_pairs_with_encoder():
    global LANG_A_TO_B, LANG_B_TO_A

    print("\n[SETUP] Use rotary encoder to choose languages.")
    print("        Rotate to change, press knob to select.\n")

    # First language (party 1)
    name_1, code_1 = choose_language_with_encoder("Select Lang 1:", LANG_CHOICES, initial_index=0)

    # Second Language (party 2)
    name_2, code_2 = choose_language_with_encoder("Select Lang 2:", LANG_CHOICES, initial_index=0)

    # Direction for Left Button
    LANG_A_TO_B = (code_1, code_2)
    LANG_B_TO_A = (code_2, code_1)

    print(f"[SETUP] Button A: {name_1} ({code_1}) → {name_2} ({code_2})")
    print(f"[SETUP] Button B: {name_2} ({code_2}) → {name_1} ({code_1})\n")

# ---------- MAIN LOOP ----------
def main():
    GPIO.setmode(PIN_MODE)
    GPIO.setup(BTN_PIN_A, GPIO.IN)
    GPIO.setup(BTN_PIN_B, GPIO.IN)

    # Encoder pins (HW-040 has its own resistors; Jetson pull_up_down is ignored anyway)
    GPIO.setup(ENC_CLK_PIN, GPIO.IN)
    GPIO.setup(ENC_DT_PIN,  GPIO.IN)
    GPIO.setup(ENC_SW_PIN,  GPIO.IN)

    setup_language_pairs_with_encoder()

    init_lcd()
    lcd_write_lines("Ready", "Hold button")

    print("Ready. Hold either button to RECORD; release to STOP + TRANSLATE.\n"
        f"- Pin {BTN_PIN_A}: {LANG_A_TO_B[0]} → {LANG_A_TO_B[1]}\n"
        f"- Pin {BTN_PIN_B}: {LANG_B_TO_A[0]} → {LANG_B_TO_A[1]}\n"
        f"- mic device: {MIC_DEVICE} @ {RATE} Hz\n"
        f"- spk device: {SPK_DEVICE}\n")


    last_a = GPIO.input(BTN_PIN_A)
    last_b = GPIO.input(BTN_PIN_B)

    # --- NEW: state for encoder SW long-press detection -----------------  # <<< NEW
    enc_sw_last = GPIO.input(ENC_SW_PIN)        # last observed level        # <<< NEW
    enc_sw_press_time = None                    # when it went LOW           # <<< NEW
    enc_sw_long_handled = False                 # to avoid repeated triggers # <<< NEW
    # ---------------------------------------------------------------------  # <<< NEW

    try:
        while True:
            state_a = GPIO.input(BTN_PIN_A)
            state_b = GPIO.input(BTN_PIN_B)

            # --- Button A logic (record / stop) ---
            if state_a != last_a:
                time.sleep(DEBOUNCE_S)
                state2 = GPIO.input(BTN_PIN_A)
                if state2 != last_a:
                    last_a = state2
                    if is_pressed(state2):
                        start_record()
                    else:
                        stop_record(*LANG_A_TO_B)

            # --- Button B logic (record / stop) ---
            if state_b != last_b:
                time.sleep(DEBOUNCE_S)
                state2 = GPIO.input(BTN_PIN_B)
                if state2 != last_b:
                    last_b = state2
                    if is_pressed(state2):
                        start_record()
                    else:
                        stop_record(*LANG_B_TO_A)

            # --- NEW: Rotary encoder SW long-press to reselect languages ---  # <<< NEW
            sw = GPIO.input(ENC_SW_PIN)

            # Edge: HIGH -> LOW (press)
            if sw == GPIO.LOW and enc_sw_last == GPIO.HIGH:
                enc_sw_press_time = time.time()
                enc_sw_long_handled = False

            # Held LOW long enough, and not currently recording
            if (
                sw == GPIO.LOW
                and enc_sw_press_time is not None
                and not enc_sw_long_handled
                and (time.time() - enc_sw_press_time) >= ENC_LONG_PRESS_S
                and rec_proc is None
            ):
                enc_sw_long_handled = True
                print("\n[LANG SEL] Long press detected on encoder SW → re-entering language selection.\n")
                lcd_write_lines("Change langs", "Rotate + press")

                # Block here while user picks new languages
                setup_language_pairs_with_encoder()

                # After re-selection, show updated mapping and ready message
                lcd_write_lines("Ready", "Hold button")

                print("Updated directions after re-selection:\n"
                      f"- Pin {BTN_PIN_A}: {LANG_A_TO_B[0]} → {LANG_A_TO_B[1]}\n"
                      f"- Pin {BTN_PIN_B}: {LANG_B_TO_A[0]} → {LANG_B_TO_A[1]}\n")

            # Edge: LOW -> HIGH (release)
            if sw == GPIO.HIGH and enc_sw_last == GPIO.LOW:
                enc_sw_press_time = None
                enc_sw_long_handled = False

            enc_sw_last = sw
            # -----------------------------------------------------------------  # <<< NEW

            lcd_scroll_step()
            time.sleep(POLL_S)

    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt")
    finally:
        if rec_proc is not None:
            stop_record(*LANG_A_TO_B)
        GPIO.cleanup()
        print("[GPIO] cleanup done")
        if LCD is not None:
            try:
                lcd_write_lines("Shutting down", "")
                time.sleep(5)
                LCD.clear()
            except Exception:
                pass

if __name__ == "__main__":
    import time
    main()
