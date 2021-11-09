import sys

from dataclasses import dataclass, field
from matplotlib.colors import from_levels_and_colors
import numpy as np
import sounddevice


@dataclass
class Measurement:
    frequency: float
    time: list[float] = field(default_factory=lambda: [])
    magnitude: list[float] = field(default_factory=lambda: [])
    state: list[int] = field(default_factory=lambda: [])


def collect_frequency(
    frequency, *, device=None, amplitude=0.5, window_size=0.05
) -> Measurement:
    high = 4000
    low = 50
    fs = 44100
    interval_time = 1.0

    state = 0
    start_idx = 0
    result = Measurement(frequency=frequency)

    def callback(indata, outdata, frames, time, status):
        nonlocal start_idx, state, result
        if status:
            print(status, file=sys.stderr)

        fft = np.fft.rfft(indata[:, 0])
        freq = np.fft.fftfreq(indata.shape[0], d=1.0 / fs)
        # Because we use rfft
        freq = freq[: fft.shape[0] - 1]
        fft = fft[: fft.shape[0] - 1]
        ind = np.logical_and(low <= freq, freq <= high)
        freq = freq[ind]
        fft = fft[ind]

        result.time.append(time.currentTime)
        db = 10.0 * np.log10(np.abs(fft))
        result.magnitude.append(np.interp(frequency, freq, db))
        result.state.append(state)

        outdata[:] = 0
        if state == 0 or state == 2:
            pass
        elif state == 1:
            t = (start_idx + np.arange(frames)) / fs
            t = t.reshape(-1, 1)
            outdata[:] = amplitude * np.sin(2 * np.pi * frequency * t)
            start_idx += frames
        else:
            raise ValueError("Unknown state")

    with sounddevice.Stream(
        device=device,
        blocksize=int(window_size * fs),
        samplerate=fs,
        channels=(1, 1),
        callback=callback,
    ):
        state = 0
        print("Recording silence...")
        sounddevice.sleep(int(interval_time * 1000))
        state = 1
        print("Recording tone...")
        sounddevice.sleep(int(interval_time * 1000))
        state = 2
        print("Recording silence...")
        sounddevice.sleep(int(2 * interval_time * 1000))

    for i in range(len(result.time) - 1, -1, -1):
        result.time[i] -= result.time[0]

    return result
