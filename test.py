import pickle
import glob
import math

import matplotlib.pyplot as plt

import reverb


def main3():
    base = 440
    freqs = [0.5 * base, 1.0 / math.sqrt(2) * base, base, math.sqrt(2) * base, 2 * base]
    results = []

    for f in freqs:
        filename = f"freq-{f}.pickle"
        try:
            result = pickle.load(open(filename, "rb"))
        except FileNotFoundError:
            result = reverb.collect_frequency(f)
            pickle.dump(result, open(filename, "wb"))
        results.append(result)

    plt.rcParams.update({"font.size": 8})
    _, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax = ax.reshape(-1)

    data_x = []
    data_y = []
    for i, result in enumerate(results):
        t = reverb.analyze(result, ax=ax[i])
        data_x.append(f"{result.frequency:.0f}")
        data_y.append(t)

    ax[-1].bar(data_x, data_y)
    ax[-1].set_xlabel("Frequency (Hz)")
    ax[-1].set_ylabel("Reverb time")
    plt.show()


main3()
