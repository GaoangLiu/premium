import matplotlib.pyplot as plt
import numpy as np

from .frame import Frame
from scipy.io.wavfile import read


class Draw(Frame):
    def __init__(self, audio_file: str = None):
        super().__init__(audio_file)

    def plot(self):
        samplerate, data = read(self._file)
        duration = len(data) / samplerate
        print("Duration of Audio in Seconds", duration)
        print("Duration of Audio in Minutes", duration / 60)

        time = np.arange(0, duration, 1 / samplerate)

        # Plotting the Graph using Matplotlib
        plt.plot(time, data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(self._file)
        plt.show()
