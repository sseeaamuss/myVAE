import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "cat-dataset/B_MIN01_EU_FN_BEN01_101.wav"

signal, sr = librosa.load(file, sr=22050) # sr * T -> 22050 * seconds of the file
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show() 

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

plt.plot(frequency[:len(magnitude)//2], magnitude[:len(magnitude)//2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# stft -> spectrogram
stft = librosa.core.stft(signal, n_fft=2048, hop_length=512)

spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log' )
plt.colorbar(img, format="%+2.0f dB")
plt.xlabel("Time (s)")
plt.ylabel("Frequency(Hz)")
plt.show()


# MFCCs
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
librosa.display.specshow(mfccs, sr=sr, hop_length=512, x_axis='time')
plt.colorbar()
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficients")
plt.show()