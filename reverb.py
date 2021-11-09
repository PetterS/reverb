import sys

import numpy as np
import sounddevice


def collect_frequency(frequency, *, device=None, amplitude=0.5, window_size=0.05):
    high = 4000
    low = 50
    fs = 44100
    interval_time = 1.0

    state = 0
    start_idx = 0
    collected_time = []
    collected_magnitude = []

    def callback(indata, outdata, frames, time, status):
        nonlocal start_idx, state, collected_time, collected_magnitude
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

        collected_time.append(time.currentTime)
        db = 10.0 * np.log10(np.abs(fft))
        collected_magnitude.append(np.interp(frequency, freq, db))

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

    for i in range(len(collected_time) - 1, -1, -1):
        collected_time[i] -= collected_time[0]

    return collected_time, collected_magnitude
