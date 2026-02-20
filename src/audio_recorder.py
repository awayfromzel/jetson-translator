"""
Audio recording abstraction using Linux arecord.

Provides start/stop control for push-to-talk recording and
writes captured audio to a WAV file for downstream processing.
"""

import os
import time
import signal
import subprocess

class AudioRecorder:
    def __init__(self, mic_device: str, rate: int, fmt: str, out_dir: str, filename: str = "speech.wav"):
        self.mic_device = mic_device
        self.rate = rate
        self.fmt = fmt
        self.out_dir = out_dir
        self.filename = filename

        self._proc = None
        self._t0 = None

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    @property
    def wav_path(self) -> str:
        return os.path.join(self.out_dir, self.filename)

    def start(self):
        if self._proc is not None:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        cmd = [
            "arecord",
            "-D", self.mic_device,
            "-f", self.fmt,
            "-c", "1",
            "-r", str(self.rate),
            "-t", "wav",
            self.wav_path,
        ]

        self._t0 = time.time()
        print(f"[REC] start → {self.wav_path}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self, timeout_s: float = 3.0) -> float:
        """
        Stops recording and returns duration in seconds.
        """
        if self._proc is None:
            return 0.0

        print("[REC] stopping…")
        self._proc.send_signal(signal.SIGINT)

        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

        dur = (time.time() - self._t0) if self._t0 else 0.0
        self._proc = None
        self._t0 = None

        print(f"[REC] done ({dur:.2f}s)")
        return dur
