# translation_pipeline.py

class TranslationPipeline:
    def __init__(self, transcribe_fn, translate_fn, tts, player, lcd, lcd_cols: int):
        self.transcribe_fn = transcribe_fn
        self.translate_fn = translate_fn
        self.tts = tts
        self.player = player
        self.lcd = lcd
        self.lcd_cols = lcd_cols

    def process(self, wav_path: str, src_lang: str, tgt_lang: str):
        self.lcd.write("Transcribing...", "")
        text = self.transcribe_fn(wav_path)
        print("[ASR]", text or "(no speech)")

        if not text:
            self.lcd.clear_scroll()
            self.lcd.write("No speech", "Try again")
            return None, None

        self.lcd.write("Translating...", "")
        out = self.translate_fn(text, src_lang, tgt_lang)
        print("[MT ]", out)

        header = f"{src_lang[:3]}→{tgt_lang[:3]}"
        self.lcd.set_scroll_text(out)
        self.lcd.write(header, out[: self.lcd_cols])

        wav = self.tts.synthesize(out, tgt_lang)
        if wav:
            self.player.play_wav(wav)

        return text, out