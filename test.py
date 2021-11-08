from numpy.fft import fftfreq
import matplotlib.pyplot as plt
import numpy as np
import sounddevice

def main():
    high = 4000
    low = 50
    fs = 44100
    duration = 2.0

    print("Recording...")
    data =sounddevice.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
    print("Done.", data.shape)

    fft = np.fft.rfft(data[:, 0])
    freq = np.fft.fftfreq(data.shape[0], d=1.0/fs)
    # Because we use rfft
    freq = freq[:fft.shape[0] - 1]
    fft = fft[:fft.shape[0] - 1]
    ind = np.logical_and(low <= freq, freq <= high)
    freq = freq[ind]
    fft = fft[ind]

    magnitude = np.abs(fft)

    plt.plot(freq, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylable("Magnitude")
    plt.show()

main()