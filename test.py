import matplotlib.pyplot as plt

import reverb


def main3():
    freq = 440
    t, m = reverb.collect_frequency(freq)

    plt.plot(t, m, "x-")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Response for {freq} Hz")
    plt.show()


main3()
