import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal):

    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show() 


def plot_fft(signal):

        # fft -> spectrum
    fft = np.fft.fft(signal)

    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))

    plt.plot(frequency[:len(magnitude)//2], magnitude[:len(magnitude)//2])
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()


def plot_spectrogram(signal,sr):

        # stft -> spectrogram
    stft = librosa.core.stft(signal, n_fft=2048, hop_length=512)

    spectrogram = np.abs(stft)

    log_spectrogram = librosa.amplitude_to_db(spectrogram)


    plt.figure(figsize=(5.12, 5.12), dpi=100)  # 5.12in × 100dpi = 512 px
    librosa.display.specshow(
        log_spectrogram,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='log',
        cmap="magma"
    )
    plt.axis("off")  # hide axes for cleaner image
    plt.show()

    


def plot_mfccs(sr, signal):

        # MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    librosa.display.specshow(mfccs, sr=sr, hop_length=512, x_axis='time')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.show()

def invert_spectrogram(spectrogram, sr, n_iter=32):
    """
    Inverts a spectrogram back to an audio signal using the Griffin-Lim algorithm.

    Args:
        spectrogram (np.ndarray): The input spectrogram (magnitude).
        sr (int): Sample rate of the audio signal.
        n_iter (int): Number of iterations for the Griffin-Lim algorithm.

    Returns:
        np.ndarray: The reconstructed audio signal.
    """
    # Use Griffin-Lim algorithm to estimate the phase and reconstruct the time-domain signal
    reconstructed_signal = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=512, win_length=2048)
    return reconstructed_signal


def main():

    file = "cat-dataset/B_MIN01_EU_FN_BEN01_101.wav"
    sr = 22050
    signal, sr = librosa.load(file, sr=sr) # sr * T -> 22050 * seconds of the file

    plot_spectrogram(signal,sr)


main()