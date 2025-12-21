# audio_player.py
import os
import subprocess
import shutil

class AudioPlayer:
    def __init__(self, spk_device: str | None, pulse_sink: str | None = None):
        self.spk_device = spk_device if spk_device else None
        self.pulse_sink = pulse_sink

    def _resolve_pulse_sink(self) -> str | None:
        """
        If pulse_sink is an exact sink name, use it.
        If it’s a substring, pick the first sink containing it.
        Return None to fall back to default sink.
        """
        if not self.pulse_sink:
            return None

        r = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return None

        sinks = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                sinks.append(parts[1])

        # Exact match
        if self.pulse_sink in sinks:
            return self.pulse_sink

        # Substring match
        for s in sinks:
            if self.pulse_sink in s:
                return s

        return None

    def play_wav(self, wav_path: str):
        env = os.environ.copy()
        print(f"[AUDIO] spk_device={self.spk_device!r} pulse_sink_cfg={self.pulse_sink!r}")
        #if self.spk_device == "pulse" and self.pulse_sink:
            # Forces the output device even if the system default changes
            #env["PULSE_SINK"] = self.pulse_sink
        if self.spk_device == "pulse":
            resolved = self._resolve_pulse_sink()
            if resolved:
                env["PULSE_SINK"] = resolved
                print(f"[AUDIO] PULSE_SINK env set to: {env.get('PULSE_SINK')!r}")
        cmd = ["aplay"]
        if self.spk_device:
            cmd += ["-D", self.spk_device]
        cmd.append(wav_path)

        #if shutil.which("pasuspender"):
            #cmd = ["pasuspender", "--"] + cmd

        if shutil.which("pasuspender") and self.spk_device != "pulse":
            cmd = ["pasuspender", "--"] + cmd

        #subprocess.run(cmd, env=env, check=False)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        print(f"[AUDIO] rc={r.returncode}")
        if r.stderr:
            print("[AUDIO] stderr:", r.stderr.strip())