import os
import numpy as np
import torch
import torchaudio
import random


keyword_labels = ["down", "go", "left", "no", "off", "on", "right",
                 "stop", "up", "yes", "silence", "unknown"]


class KeywordSpottingDataset(torchaudio.datasets.SPEECHCOMMANDS):
    """TODO: make sub selection of labels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.augment = kwargs['subset'] == 'training'
        self.load_background_noise_samples()
        self.prepare_label_subset()
        self.input_length = 16000

    def load_background_noise_samples(self):
        """Load background audio samples for data augmentation
        """
        background_noise_path = os.path.join(self._path, '_background_noise_')
        self.background_noise_samples = []
        for f in os.listdir(background_noise_path):
            if f.endswith('.wav'):
                filename = os.path.join(background_noise_path, f)
                self.background_noise_samples.append(torchaudio.load(filename)[0])

    def prepare_label_subset(self):
        """We don't use all 35 labels from the dataset, but only a subset of labels `keyword_labels`.
        In addition, we add a 'silence' and 'unknown' label, where the latter one contains
        samples from the unused labels. We make sure that the amount of samples of the two new labels
        more or less equals that of the other labels to keep the balance.
        """
        # separate known from unknown labels
        samples_known_label = []
        samples_unknown_label = []
        for w in self._walker:
            label = w.split(os.sep)[-2]
            if label in keyword_labels:
                samples_known_label.append((w, keyword_labels.index(label)))
            else:
                samples_unknown_label.append((w, keyword_labels.index('unknown')))

        # compose a random selection of unknown labels, comparable to the amount of samples
        # for a known label
        rng = np.random.RandomState(0)   # deterministic random generator to always select the same random subset
        rng.shuffle(samples_unknown_label)
        avg_num_samples_per_label = len(samples_known_label) // (len(keyword_labels) - 2)   # -2 for the unknown and silence label
        samples_unknown_label = samples_unknown_label[:avg_num_samples_per_label]

        # generate and equal amount of 'silence' samples
        samples_silence = [(None, keyword_labels.index('silence'))] * avg_num_samples_per_label

        self.data = samples_known_label + samples_unknown_label + samples_silence

    def random_time_shift(self, data, shift_ms=100):
        shift = (data.shape[1] * shift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = torch.nn.functional.pad(data, (a, b), "constant")
        return data[:,:data.shape[1] - a] if a else data[:,b:]

    def random_bg_noise(self, data):
        """Mix a random background noise with the data sample
        """
        bg_noise = random.choice(self.background_noise_samples)

        # select random clip from long background noise sample
        a = random.randint(0, bg_noise.shape[1] - data.shape[1] - 1)
        bg_noise = bg_noise[:,a:a + data.shape[1]]

        # mix background noise with keyword clip
        a = random.random() * 0.1
        data = torch.clip(a * bg_noise + data, -1, 1)

        return data

    def to_mfcc(self, data):
        """
        Calculate MFCC spectogram (Mel Frequency Cepstral Coefficient):
            * Create spectogram (STFT = Short-Time Fourier Transform) -> FFT for every 30 ms frame
            * Apply Mel-spaced filterbanks to get Mel spectogram -> 40 narrow mel filters
            * Take the log to get log-mel spectogram
            * Take the DCT (Discrete Cosine Transform) to get the final cepstral coefficients -> 10 coefficients

        Parameters below are taken from tensorflow implementation
        """
        transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=10,
            log_mels=True,
            melkwargs={"n_fft": 480,
                       "hop_length": 330,   # originally 320 in TF, but due to implementation difference,
                                            # we use 330 to get the same 49 (as in TF framework) time steps in the final tensor
                       "n_mels": 40,
                       "power": 1,
                       "f_min": 20.0,
                       "f_max": 4000},
        )
        return transform(data)

    def __getitem__(self, index):
        sample_path, label = self.data[index]
        if sample_path is None:
            data1 = torch.zeros(1, self.input_length)
        else:
            data, _ = torchaudio.load(sample_path)
            # add zero padding to the end if sample is shorter than the expected length
            data1 = torch.nn.functional.pad(data, (0, max(0, self.input_length - data.shape[1])), "constant")

        data3 = data1
        if self.augment:
            data2 = self.random_time_shift(data1)
            data3 = self.random_bg_noise(data2)

        # convert to MFCC
        data = self.to_mfcc(data3)

        return data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt

    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
        plt.show()

    train_dataset = KeywordSpottingDataset(
        "./data/",
        download=True,
        subset="training"
    )

    val_dataset = KeywordSpottingDataset(
        "./data/",
        download=True,
        subset="validation"
    )

    plot_spectrogram(train_dataset[0][0][0], 'mfcc')
