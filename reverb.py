import argparse
import math
import os
import pickle
import sys
from typing import Union

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import sounddevice


@dataclass
class Measurement:
    frequency: float
    time: Union[np.array, list[float]] = field(default_factory=lambda: [])
    magnitude: Union[np.array, list[float]] = field(default_factory=lambda: [])
    state: Union[np.array, list[int]] = field(default_factory=lambda: [])


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
        print("-- Recording silence...")
        sounddevice.sleep(int(interval_time * 1000))
        state = 1
        print(f"-- Recording tone of {frequency:.0f} Hz...")
        sounddevice.sleep(int(interval_time * 1000))
        state = 2
        print("-- Recording silence...")
        sounddevice.sleep(int(2 * interval_time * 1000))

    for i in range(len(result.time) - 1, -1, -1):
        result.time[i] -= result.time[0]

    result.time = np.array(result.time)
    result.magnitude = np.array(result.magnitude)
    result.state = np.array(result.state, dtype=np.int32)
    return result


def analyze(measurement: Measurement, ax=None):
    label = f"{measurement.frequency} Hz"
    robust_max = np.partition(measurement.magnitude, -2)[-2]

    max_ind = np.nonzero(measurement.magnitude >= robust_max - 1.5)[0]
    max_level = np.median(measurement.magnitude[max_ind])

    i = max_ind[0] - 4
    min_level = np.max(measurement.magnitude[range(0, i)])

    if ax:
        p = ax.plot(measurement.time, measurement.magnitude, ".-", label=label)
        ax.plot(
            measurement.time[max_ind],
            measurement.magnitude[max_ind],
            "o",
            color=p[0].get_color(),
        )
        ax.plot(
            [measurement.time[0], measurement.time[-1]],
            [max_level, max_level],
            ":",
            color="k",
        )
        ax.plot(
            [measurement.time[0], measurement.time[-1]],
            [min_level, min_level],
            ":",
            color="k",
        )

    start_decline_i = max_ind[-1]
    start_decline_t = measurement.time[start_decline_i]

    last_big_i = np.nonzero(measurement.magnitude > min_level + 5.0)[0][-1]
    last_big_t = measurement.time[last_big_i]
    end_decline_i = np.nonzero(
        np.logical_and(
            measurement.time > last_big_t, measurement.magnitude < min_level + 1.0
        )
    )[0][0]
    end_decline_t = measurement.time[end_decline_i]

    dt = measurement.time[start_decline_i : end_decline_i + 1]
    dm = measurement.magnitude[start_decline_i : end_decline_i + 1]
    if True:
        A = np.ones((len(dt), 1))
        lin_k = (dm[-1] - dm[0]) / (dt[-1] - dt[0])
        lin_m = dm[0] - lin_k * dt[0]
    else:
        A = np.vstack([dt, np.ones(len(dt))]).T
        lin_k, lin_m = np.linalg.lstsq(A, dm, rcond=None)[0]

    RT60 = -60.0 / lin_k

    if ax:
        yl = ax.get_ylim()
        tt = np.array([start_decline_t, end_decline_t])
        ax.plot([tt[0], tt[0]], yl, "-", color="k")
        ax.plot([tt[1], tt[1]], yl, "-", color="k")
        ax.plot(tt, lin_k * tt + lin_m, "-", color="r")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Response for {measurement.frequency:.0f} Hz. RT60={RT60:.2f} s")

    return RT60


def main():
    parser = argparse.ArgumentParser(description="Calculate RT60 reverberation.")
    parser.add_argument("--room", required=True, help="The name of the room to measure")
    args = parser.parse_args()

    base = 440
    freqs = [
        0.5 * base,
        1.0 / math.sqrt(2) * base,
        base,
        math.sqrt(2) * base,
        2 * base,
        3 * base,
        4 * base,
    ]
    results = []

    for f in freqs:
        filename = f"data/{args.room}/freq-{f:.0f}.pickle"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            result = pickle.load(open(filename, "rb"))
            print(f"Have result for {f:.0f} Hz")
        except FileNotFoundError:
            print(f"Collecting result for {f:.0f} Hz...")
            result = collect_frequency(f)
            pickle.dump(result, open(filename, "wb"))
        results.append(result)

    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax = ax.reshape(-1)

    data_x = []
    data_y = []
    for i, result in enumerate(results):
        try:
            t = analyze(result, ax=ax[i])
            data_x.append(f"{result.frequency:.0f}")
            data_y.append(t)
        except Exception as e:
            print(e)

    ax[-1].bar(data_x, data_y)
    ax[-1].set_xlabel("Frequency (Hz)")
    ax[-1].set_ylabel("Reverb time")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
