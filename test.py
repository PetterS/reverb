import matplotlib.pyplot as plt

import reverb


def main3():
    freq = 440
    result = reverb.collect_frequency(freq)

    plt.plot(result.time, result.magnitude, "x-")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Response for {freq} Hz")

    plt.plot(result.time, result.state)
    plt.show()


main3()
