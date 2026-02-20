"""
Application assembly and dependency wiring.

Initializes hardware interfaces, loads ML models (ASR + MT),
configures the TTS backend client, and constructs the
AppController with all required dependencies.

Acts as the composition root of the system.
"""

import os
import re
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lcd_display import LCDDisplay
from audio_recorder import AudioRecorder
from audio_player import AudioPlayer
#from tts_backend import PiperSubprocessTTS
from tts_backend_http import PiperHTTPServiceTTS
#from tts_http_client import PiperHttpTTS
from translation_pipeline import TranslationPipeline
from gpio_inputs import DebouncedButton, LongPressDetector
from language_selector import LanguageSelector
from app_controller import AppController
import wave

def _ensure_silence_wav(path: str, rate: int, seconds: float = 0.35):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return

    nframes = int(rate * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit PCM
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * nframes)  # silence

def _split_sentences(text: str):
    """
    Split into sentence-like chunks, keeping punctuation (?, !, .) attached.
    Example:
      "Hi? I'm fine." -> ["Hi?", "I'm fine."]
    """
    text = (text or "").strip()
    if not text:
        return []

    # Minimal match up to ., ?, ! (one or more), or end of string.
    chunks = re.findall(r".+?(?:[.!?]+(?=\s|$)|$)", text, flags=re.DOTALL)
    return [c.strip() for c in chunks if c.strip()]


def build_controller(GPIO, config, audio_out_dir: str):
    # ----- LCD -----
    lcd = LCDDisplay(
        cols=config["LCD_COLS"],
        rows=config["LCD_ROWS"],
        i2c_address=config["LCD_I2C_ADDR"],
        i2c_port=config["LCD_I2C_PORT"],
        charmap=config["LCD_CHARMAP"],
        auto_linebreaks=config["LCD_AUTO_LINEBREAKS"],
        scroll_interval_s=config["SCROLL_INTERVAL"],
    )
    lcd.init()
    lcd.write("Loading...", "ASR model")

    # ----- paths -----
    base_dir = config["BASE_DIR"]
    project_root = config["PROJECT_ROOT"]
    audio_out_dir = os.path.join(project_root, "audio_files")
    piper_voice_dir = os.path.join(project_root, "piper_voices")

    # ----- audio recorder -----
    recorder = AudioRecorder(
        mic_device=config["MIC_DEVICE"],
        rate=config["RATE"],
        fmt=config["FORMAT"],
        out_dir=audio_out_dir,
        filename="speech.wav",
    )

    # ----- player + TTS -----
    #player = AudioPlayer(spk_device=config["SPK_DEVICE"])

    player = AudioPlayer(
        spk_device=config["SPK_DEVICE"],
        pulse_sink=config.get("PULSE_SINK"),
    )


    tts = PiperHTTPServiceTTS(
    base_url="http://127.0.0.1:5005",
    out_dir=audio_out_dir,
)
    '''
    tts = PiperSubprocessTTS(
        piper_bin=config["PIPER_BIN"],
        voice_dir=piper_voice_dir,
        out_dir=audio_out_dir,
        voice_map=config["VOICE_MAP"],
        spk_device=config["SPK_DEVICE"],
    )
    '''

    # ----- ASR -----
    asr = None
    for dev, ctype in [("cuda", "int8_float16"), ("cuda", "float16"), ("cpu", "int8")]:
        try:
            asr = WhisperModel("small", device=dev, compute_type=ctype, device_index=0)
            print(f"[ASR] using {dev}/{ctype}")
            break
        except Exception as e:
            print(f"[ASR] {dev}/{ctype} failed: {e}")
    if asr is None:
        raise RuntimeError("Failed to initialize faster-whisper in all modes.")

    def transcribe(path: str) -> str:
        segs, _info = asr.transcribe(path, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()

    # ----- MT -----
    lcd.write("Loading...", "MT model")
    tok = AutoTokenizer.from_pretrained(config["MODEL_ID"])
    mt = AutoModelForSeq2SeqLM.from_pretrained(
        config["MODEL_ID"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        mt = mt.to("cuda")

    def lang_id(code: str) -> int:
        if hasattr(tok, "lang_code_to_id") and isinstance(tok.lang_code_to_id, dict) and code in tok.lang_code_to_id:
            return tok.lang_code_to_id[code]
        tid = tok.convert_tokens_to_ids(code)
        if tid is not None and tid != tok.unk_token_id:
            return tid
        if hasattr(tok, "lang_code_to_token"):
            return tok.convert_tokens_to_ids(tok.lang_code_to_token[code])
        raise ValueError(f"Unknown language code: {code}")
    '''
    def translate(text: str, src: str, tgt: str, max_new: int = 200) -> str:
        inputs = tok(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        gen = mt.generate(**inputs, max_new_tokens=max_new, forced_bos_token_id=lang_id(tgt))
        return tok.batch_decode(gen, skip_special_tokens=True)[0]
    '''

    '''
    def translate(text: str, src: str, tgt: str, max_new: int = 200) -> str:
        # IMPORTANT: Tell NLLB what language the INPUT text is in
        if hasattr(tok, "src_lang"):
            tok.src_lang = src
        elif hasattr(tok, "set_src_lang_special_tokens"):
            tok.set_src_lang_special_tokens(src)

        # (Optional but harmless) Some tokenizers also track target language
        if hasattr(tok, "tgt_lang"):
            tok.tgt_lang = tgt

        inputs = tok(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        gen = mt.generate(
            **inputs,
            max_new_tokens=max_new,
            forced_bos_token_id=lang_id(tgt),  # this sets the OUTPUT language
            num_beams = 4,
        )
        return tok.batch_decode(gen, skip_special_tokens=True)[0]
    '''

    def translate(text: str, src: str, tgt: str, max_new: int = 200) -> str:
        # IMPORTANT: Tell NLLB what language the INPUT text is in
        if hasattr(tok, "src_lang"):
            tok.src_lang = src
        elif hasattr(tok, "set_src_lang_special_tokens"):
            tok.set_src_lang_special_tokens(src)

        # (Optional but harmless) Some tokenizers also track target language
        if hasattr(tok, "tgt_lang"):
            tok.tgt_lang = tgt

        # ---- sentence-by-sentence translation ----
        parts = _split_sentences(text)
        if not parts:
            return ""

        out_parts = []
        for part in parts:
            inputs = tok(part, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            gen = mt.generate(
                **inputs,
                max_new_tokens=max_new,
                forced_bos_token_id=lang_id(tgt),
                num_beams=4,   # keep your better quality setting
            )
            out_parts.append(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())

        return " ".join([p for p in out_parts if p])


    if config.get("WARMUP", True):
        # ----- warm-up (avoids slow first real run) -----
        try:
            warm_path = os.path.join(audio_out_dir, "_warmup_silence.wav")
            _ensure_silence_wav(warm_path, rate=config["RATE"], seconds=0.35)

            lcd.write("Warming up...", "ASR")
            _ = transcribe(warm_path)

            # Warm MT with a real pair (pick first two choices from your config)
            src0 = config["LANG_CHOICES"][0][1]
            tgt0 = config["LANG_CHOICES"][1][1] if len(config["LANG_CHOICES"]) > 1 else src0

            lcd.write("Warming up...", "MT")
            _ = translate("hello", src0, tgt0)

            # (Optional) warm TTS too:
            # lcd.write("Warming up...", "TTS")
            # _ = tts.synthesize("hello", tgt0)

        except Exception as e:
            print(f"[WARMUP] skipped due to error: {e}")

    # ----- pipeline -----
    pipeline = TranslationPipeline(
        transcribe_fn=transcribe,
        translate_fn=translate,
        tts=tts,
        player=player,
        lcd=lcd,
        lcd_cols=config["LCD_COLS"],
    )

    # ----- GPIO setup -----
    GPIO.setmode(config["PIN_MODE"])
    GPIO.setup(config["BTN_PIN_A"], GPIO.IN)
    GPIO.setup(config["BTN_PIN_B"], GPIO.IN)
    GPIO.setup(config["ENC_CLK_PIN"], GPIO.IN)
    GPIO.setup(config["ENC_DT_PIN"], GPIO.IN)
    GPIO.setup(config["ENC_SW_PIN"], GPIO.IN)

    selector = LanguageSelector(
        gpio=GPIO,
        lcd=lcd,
        clk_pin=config["ENC_CLK_PIN"],
        dt_pin=config["ENC_DT_PIN"],
        sw_pin=config["ENC_SW_PIN"],
        debounce_s=config["DEBOUNCE_S"],
        poll_s=config["POLL_S"],
        sw_active_low=True,
    )

    btn_a = DebouncedButton(config["BTN_PIN_A"], gpio=GPIO, active_high=config["ACTIVE_HIGH"], debounce_s=config["DEBOUNCE_S"])
    btn_b = DebouncedButton(config["BTN_PIN_B"], gpio=GPIO, active_high=config["ACTIVE_HIGH"], debounce_s=config["DEBOUNCE_S"])

    lang_longpress = LongPressDetector(
        config["ENC_SW_PIN"],
        gpio=GPIO,
        long_press_s=config["ENC_LONG_PRESS_S"],
        active_high=False,
        debounce_s=config["DEBOUNCE_S"],
    )

    # ----- controller -----
    controller = AppController(
        gpio=GPIO,
        lcd=lcd,
        recorder=recorder,
        pipeline=pipeline,
        btn_a=btn_a,
        btn_b=btn_b,
        lang_longpress=lang_longpress,
        selector=selector,
        lang_choices=config["LANG_CHOICES"],
        pin_btn_a=config["BTN_PIN_A"],
        pin_btn_b=config["BTN_PIN_B"],
        mic_device=config["MIC_DEVICE"],
        spk_device=config["SPK_DEVICE"],
        rate_hz=config["RATE"],
        poll_s=config["POLL_S"],
    )
    return controller
