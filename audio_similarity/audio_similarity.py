import numpy as np
import librosa
import pystoi
import os
import matplotlib.pyplot as plt
import warnings
import sys
from functools import lru_cache
from tqdm import tqdm
import logging
import random

warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

logger = logging.getLogger()

class AudioSimilarity:
    """
    Calculate the similarity between audio files using multiple audio similarity metrics.

    Args:
        original_path (str): Path to the original audio file or directory.
        compare_path (str): Path to the compare audio file or directory.
        sample_rate (int): Target sample rate for audio resampling.
        weights (dict or list): Weights for the audio similarity metrics. Default is None.
        verbose (bool): Flag to enable/disable verbose logging. Default is True.
        sample_size (int): Number of audio files to sample from directories. Default is 1.

    Raises:
        None

    Notes:
        The AudioSimilarity class calculates the similarity between audio files using multiple audio similarity metrics.
        It supports various metrics including zero-crossing rate (ZCR) similarity, rhythm similarity, spectral flux similarity,
        energy envelope similarity, spectral contrast similarity, and perceptual similarity. The class can handle both
        individual audio files and directories of audio files. The loaded audio files are resampled to the specified sample
        rate for consistent processing. The weights parameter allows customization of the importance of each similarity metric.
        By default, equal weights are assigned to all metrics. Alternatively, weights can be provided as a dictionary or list.
        If a sample size is specified, a random subset of audio files is sampled from the directories.

    """
    def __init__(self, original_path, compare_path, sample_rate, weights=None, verbose=True, sample_size=1):
        
        log_format = "%(message)s"
        logging.basicConfig(level=logging.INFO if verbose else logging.CRITICAL, format=log_format)

        if weights is None:
            self.weights = {
                'zcr_similarity': 0.2,
                'rhythm_similarity': 0.2,
                'spectral_flux_similarity': 0.2,
                'energy_envelope_similarity': 0.1,
                'spectral_contrast_similarity': 0.1,
                'perceptual_similarity': 0.2
            }
        else:
            self.weights = self.parse_weights(weights)

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.original_path = original_path
        self.compare_path = compare_path

        # Check if the paths are directories or files
        self.is_directory = [os.path.isdir(path) for path in [original_path, compare_path]]

        # Load audio files once
        self.original_audios, self.compare_audios = self.load_audio_files()
        
        # Check if valid audio files were loaded
        if not self.original_audios or not self.compare_audios:
            sys.exit("No valid audio files found in the provided paths.")

    def parse_weights(self, weights):
        if isinstance(weights, dict):
            return weights
        elif isinstance(weights, list):
            if len(weights) != 6:
                raise ValueError("Invalid number of weights. Expected 6 weights.")
            metric_names = [
                'zcr_similarity',
                'rhythm_similarity',
                'spectral_flux_similarity',
                'energy_envelope_similarity',
                'spectral_contrast_similarity',
                'perceptual_similarity'
            ]
            return dict(zip(metric_names, weights))
        else:
            raise ValueError("Invalid type for weights. Expected dict or list.")


    def load_audio_files(self):
        """
        Load and preprocess the original and compare audio files.

        Returns:
            tuple: A tuple containing two lists of loaded audio files - original_audios and compare_audios.

        Raises:
            None

        Notes:
            This method loads the audio files from the specified paths and performs preprocessing steps.
            If the paths represent directories, it loads all audio files with valid extensions from the directories.
            If the paths represent individual audio files, it loads those files.
            The loaded audio files are stored in separate lists - original_audios and compare_audios.
            The audio files are randomly sampled if a sample size is specified.
            The loaded audio files are preprocessed using librosa, including resampling to the specified sample rate.

        """
        original_audios = []
        compare_audios = []
        valid_extensions = ('.mp3', '.flac', '.wav')

        if self.is_directory[0]:
            try:
                original_files = [os.path.join(self.original_path, f) for f in os.listdir(self.original_path) if f.endswith(valid_extensions)]
            except FileNotFoundError as e:
                logging.error(f"Error loading original audio files: {e}")
                return [], []
        else:
            if not os.path.isfile(self.original_path):
                logging.error(f"Invalid original file path: {self.original_path}")
                return [], []
            original_files = [self.original_path]

        if self.is_directory[1]:
            try:
                compare_files = [os.path.join(self.compare_path, f) for f in os.listdir(self.compare_path) if f.endswith(valid_extensions)]
            except FileNotFoundError as e:
                logging.error(f"Error loading compare audio files: {e}")
                return [], []
        else:
            if not os.path.isfile(self.compare_path):
                logging.error(f"Invalid compare file path: {self.compare_path}")
                return [], []
            compare_files = [self.compare_path]

        if not original_files:
            logging.error("No original audio files found.")
        if not compare_files:
            logging.error("No compare audio files found.")

        # Randomly sample files from the original files list
        original_files = random.sample(original_files, self.sample_size) if self.sample_size else original_files

        # Randomly sample files from the compare files list
        compare_files = random.sample(compare_files, self.sample_size) if self.sample_size else compare_files

        for original_file in tqdm(original_files, desc="Loading original files:"):
            try:
                original_audio, _ = librosa.load(original_file, sr=self.sample_rate)
                original_audios.append(original_audio)
            except FileNotFoundError as e:
                logging.error(f"Error loading file {original_file}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error loading file {original_file}: {type(e).__name__}, {e}")
                continue

        for compare_file in tqdm(compare_files, desc="Loading comparison files:"):
            try:
                compare_audio, _ = librosa.load(compare_file, sr=self.sample_rate)
                compare_audios.append(compare_audio)
            except FileNotFoundError as e:
                logging.error(f"Error loading file {compare_file}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error loading file {compare_file}: {type(e).__name__}, {e}")
                continue

        return original_audios, compare_audios


    @lru_cache(maxsize=None)
    def zcr_similarity(self):
        """
        Calculate the zero-crossing rate (ZCR) similarity between audio files.

        Returns:
            float: The average ZCR similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The ZCR similarity is calculated by comparing the zero-crossing rates of the audio signals.
            Zero-crossing rate represents the rate at which the audio signal changes its sign. The similarity
            score is obtained by calculating the absolute difference between the ZCR of the original and compare
            audio files and subtracting it from 1. The similarity score ranges between 0 and 1, where a higher
            score indicates greater similarity.

        """
        logging.info("Calculating zero crossing rate similarity...")

        total_zcr_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_zcr = np.mean(np.abs(np.diff(np.sign(original_audio))) > 0)

            for compare_audio in self.compare_audios:
                compare_zcr = np.mean(np.abs(np.diff(np.sign(compare_audio))) > 0)
                zcr_similarity = 1 - np.abs(original_zcr - compare_zcr)
                total_zcr_similarity += zcr_similarity
                count += 1

        if count > 0:
            return total_zcr_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def rhythm_similarity(self):
        """
        Calculate the rhythm similarity between audio files.

        Returns:
            float: The average rhythm similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The rhythm similarity is calculated by comparing the rhythm patterns of the audio signals.
            Rhythm patterns are derived from the onsets in the audio. The similarity score is obtained
            by calculating the Pearson correlation coefficient between the rhythm patterns of the original
            and compare audio files and normalizing it between 0 and 1. The similarity score ranges between
            0 and 1, where a higher score indicates greater similarity.

    """
        logging.info("Calculating rhythm similarity...")
        total_rhythm_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            min_length = len(original_audio)
            original_onset_vector = np.zeros(min_length)
            original_onsets = librosa.onset.onset_detect(y=original_audio, sr=self.sample_rate, units='time')
            original_onsets = np.array(original_onsets) * self.sample_rate
            original_onsets = original_onsets[original_onsets < min_length]
            original_onset_vector[original_onsets.astype(int)] = 1

            for compare_audio in self.compare_audios:
                min_length = min(min_length, len(compare_audio))
                compare_onset_vector = np.zeros(min_length)
                compare_onsets = librosa.onset.onset_detect(y=compare_audio[:min_length], sr=self.sample_rate, units='time')
                compare_onsets = np.array(compare_onsets) * self.sample_rate
                compare_onsets = compare_onsets[compare_onsets < min_length]
                compare_onset_vector[compare_onsets.astype(int)] = 1

                rhythm_similarity = (np.corrcoef(original_onset_vector[:min_length], compare_onset_vector[:min_length])[0, 1] + 1) / 2
                total_rhythm_similarity += rhythm_similarity
                count += 1

        if count > 0:
            return total_rhythm_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def chroma_similarity(self):
        """
        Calculate the chroma similarity between audio files.

        Returns:
            float: The average chroma similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The chroma similarity is calculated by comparing the chroma features of the audio signals.
            Chroma features represent the distribution of pitches in the audio. The similarity score is
            obtained by calculating the mean absolute difference between the chroma features of the original
            and compare audio files, and subtracting it from 1. The similarity score ranges between 0 and 1,
            where a higher score indicates greater similarity.

        """
        logging.info("Calculating chroma similarity similarity...")
        total_chroma_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            for compare_audio in self.compare_audios:
                original_chroma = librosa.feature.chroma_cqt(y=original_audio, sr=self.sample_rate)
                compare_chroma = librosa.feature.chroma_cqt(y=compare_audio, sr=self.sample_rate)

                min_length = min(original_chroma.shape[1], compare_chroma.shape[1])
                original_chroma = original_chroma[:, :min_length]
                compare_chroma = compare_chroma[:, :min_length]

                chroma_similarity = 1 - np.mean(np.abs(original_chroma - compare_chroma))
                total_chroma_similarity += chroma_similarity
                count += 1

        if count > 0:
            return total_chroma_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None


    @lru_cache(maxsize=None)
    def energy_envelope_similarity(self):
        """
        Calculate the energy envelope similarity between audio files.

        Returns:
            float: The average energy envelope similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The energy envelope similarity is calculated by comparing the energy envelopes of the audio signals.
            The energy envelope represents the variation of the signal's energy over time. The similarity score
            is obtained by calculating the Pearson correlation coefficient between the energy envelopes of the
            original and compare audio files and normalizing it between 0 and 1. The similarity score ranges between
            0 and 1, where a higher score indicates greater similarity.

        """
        logging.info("Calculating energy envelope similarity...")
        total_energy_envelope_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            min_length = len(original_audio)
            original_energy_envelope = np.abs(original_audio[:min_length])

            for compare_audio in self.compare_audios:
                min_length = min(min_length, len(compare_audio))
                compare_energy_envelope = np.abs(compare_audio[:min_length])

                energy_envelope_similarity = (np.corrcoef(original_energy_envelope[:min_length], compare_energy_envelope[:min_length])[0, 1] + 1) / 2
                total_energy_envelope_similarity += energy_envelope_similarity
                count += 1

        if count > 0:
            return total_energy_envelope_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def spectral_contrast_similarity(self):
        """
        Calculate the spectral contrast similarity between audio files.

        Returns:
            float: The average spectral contrast similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The spectral contrast similarity is calculated by comparing the spectral contrast of the audio signals.
            Spectral contrast measures the difference in magnitudes between peaks and valleys in the spectrum,
            representing the perceived amount of spectral emphasis. The spectral contrast similarity score is
            obtained by comparing the spectral contrast of the original and compare audio files and calculating
            the average normalized similarity. The similarity score ranges between 0 and 1, where a higher score
            indicates greater similarity.

        """
        logging.info("Calculating spectral contrast similarity...")
        if not self.original_audios or not self.compare_audios:
            logging.info("No audio files loaded.")
            return None

        total_spectral_contrast_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_contrast = librosa.feature.spectral_contrast(y=original_audio, sr=self.sample_rate)
            min_columns = original_contrast.shape[1]

            for compare_audio in self.compare_audios:
                compare_contrast = librosa.feature.spectral_contrast(y=compare_audio, sr=self.sample_rate)
                min_columns = min(min_columns, compare_contrast.shape[1])

            original_contrast = original_contrast[:, :min_columns]

            for compare_audio in self.compare_audios:
                compare_contrast = librosa.feature.spectral_contrast(y=compare_audio, sr=self.sample_rate)
                compare_contrast = compare_contrast[:, :min_columns]
                contrast_similarity = np.mean(np.abs(original_contrast - compare_contrast))
                normalized_similarity = 1 - contrast_similarity / np.max([np.abs(original_contrast), np.abs(compare_contrast)])
                total_spectral_contrast_similarity += normalized_similarity
                count += 1

        if count > 0:
            return total_spectral_contrast_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def perceptual_similarity(self, sr=16000):
        """
        Calculate the perceptual similarity between audio files using the Short-Time Objective Intelligibility (STOI) metric.

        Args:
            sr (int): Target sample rate for resampling the audio files. Default is 16000.

        Returns:
            float: The average perceptual similarity score between the audio files, normalized between 0 and 1.

        Raises:
            None

        Notes:
            The perceptual similarity is calculated using the Short-Time Objective Intelligibility (STOI) metric.
            STOI measures the similarity of two audio signals in terms of their intelligibility. The STOI score
            ranges between -1 and 1, where a higher score indicates greater similarity. The perceptual similarity
            score is obtained by normalizing the STOI score between 0 and 1, where 0 indicates no similarity and 1
            indicates perfect similarity.

        """
        logging.info("Calculating perceptual similarity...")
        if not self.original_audios or not self.compare_audios:
            logging.info("No audio files loaded.")
            return None

        total_perceptual_similarity = 0
        count = 0

        for i, original_audio in enumerate(self.original_audios):
            for j, compare_audio in enumerate(self.compare_audios):
                min_length = min(len(original_audio), len(compare_audio))
                array1_16k = librosa.resample(y=original_audio[:min_length], orig_sr=self.sample_rate, target_sr=sr)
                array2_16k = librosa.resample(y=compare_audio[:min_length], orig_sr=self.sample_rate, target_sr=sr)
                score = pystoi.stoi(array1_16k, array2_16k, sr)
                score_normalized = (score + 1) / 2
                total_perceptual_similarity += score_normalized
                count += 1

        if count > 0:
            return total_perceptual_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None


    def stent_weighted_audio_similarity(self, metrics='swass'):
        """
        Calculate the Stent Weighted Audio Similarity Score (SWASS) and other audio similarity metrics.

        Args:
            metrics (str): Type of metrics to calculate. Choose 'swass' for the Stent Weighted Audio Similarity Score,
                or 'all' to calculate all available audio similarity metrics. Default is 'swass'.

        Returns:
            If metrics is 'swass':
                float: The Stent Weighted Audio Similarity Score (SWASS) normalized between 0 and 1.
            If metrics is 'all':
                dict: A dictionary containing the calculated audio similarity metrics, including:
                    - 'zcr_similarity': The average Zero-Crossing Rate (ZCR) similarity between the audio files.
                    - 'rhythm_similarity': The average rhythm similarity between the audio files.
                    - 'chroma_similarity': The average chroma similarity between the audio files.
                    - 'energy_envelope_similarity': The average energy envelope similarity between the audio files.
                    - 'spectral_contrast_similarity': The average spectral contrast similarity between the audio files.
                    - 'perceptual_similarity': The average perceptual similarity between the audio files.
                    - 'swass': The Stent Weighted Audio Similarity Score (SWASS) normalized between 0 and 1.

        Raises:
            None

        Notes:
            The Stent Weighted Audio Similarity Score (SWASS) measures the overall similarity between two sets of audio files
            based on multiple audio similarity metrics. It takes into account the weights assigned to each metric to calculate
            a weighted average similarity score. The SWASS value ranges between 0 and 1, where 0 indicates no similarity and 1
            indicates perfect similarity.

        """
        if not self.original_audios or not self.compare_audios:
            logging.error("No audio files loaded.")
            return None

        num_original_audios = len(self.original_audios)
        num_compare_audios = len(self.compare_audios)

        zcr_similarities = np.zeros((num_original_audios, num_compare_audios))
        rhythm_similarities = np.zeros((num_original_audios, num_compare_audios))
        chroma_similarity = np.zeros((num_original_audios, num_compare_audios))
        energy_envelope_similarities = np.zeros((num_original_audios, num_compare_audios))
        spectral_contrast_similarities = np.zeros((num_original_audios, num_compare_audios))
        perceptual_similarities = np.zeros((num_original_audios, num_compare_audios))

        for i, original_audio in enumerate(self.original_audios):
            self.original_audio = original_audio
            for j, compare_audio in enumerate(self.compare_audios):
                self.compare_audio = compare_audio
                zcr_similarities[i, j] = self.zcr_similarity()
                rhythm_similarities[i, j] = self.rhythm_similarity()
                chroma_similarity[i, j] = self.chroma_similarity()
                energy_envelope_similarities[i, j] = self.energy_envelope_similarity()
                spectral_contrast_similarities[i, j] = self.spectral_contrast_similarity()
                perceptual_similarities[i, j] = self.perceptual_similarity()

        weights = np.array(list(self.weights.values()))
        similarities = (
            weights[0] * zcr_similarities +
            weights[1] * rhythm_similarities +
            weights[2] * chroma_similarity +
            weights[3] * energy_envelope_similarities +
            weights[4] * spectral_contrast_similarities +
            weights[5] * perceptual_similarities
        )

        total_similarity = np.sum(similarities)
        count = num_original_audios * num_compare_audios

        if metrics == 'all':
            return {
                'zcr_similarity': float(np.mean(zcr_similarities)),
                'rhythm_similarity': float(np.mean(rhythm_similarities)),
                'chroma_similarity': float(np.mean(chroma_similarity)),
                'energy_envelope_similarity': float(np.mean(energy_envelope_similarities)),
                'spectral_contrast_similarity': float(np.mean(spectral_contrast_similarities)),
                'perceptual_similarity': float(np.mean(perceptual_similarities)),
                'swass': float(total_similarity / count)
            }
        elif metrics == 'swass':
            if count > 0:
                return float(total_similarity / count)
            else:
                logging.error("No audio files loaded.")
                return None
        else:
            logging.error("Invalid value for 'metrics'. Choose 'swass' or 'all'.")
            return None


    def plot(self, metrics=None, option='radar', figsize=(8, 6), alpha=0.5, title=None, dpi=300, savefig=False, fontsize=12, label_fontsize=10, title_fontsize=14, color1='blue', color2='green'):
        """
        Plot function for visualizing metrics.

        Args:
            metrics (list): List of metrics to plot. If None, default metrics will be used.
            option (str): Plotting option. Choose from 'radar', 'bar', or 'all'.
            figsize (tuple): Figure size in inches. Default is (8, 6).
            alpha (float): Transparency of the plotted elements. Default is 0.5.
            title (str): Title of the plot. Default is None.
            dpi (int): Dots per inch of the figure. Default is 300.
            savefig (bool): Flag indicating whether to save the plot as an image file. Default is False.
            fontsize (int): Font size for the plot. Default is 12.
            label_fontsize (int): Font size for tick labels. Default is 10.
            title_fontsize (int): Font size for the title. Default is 14.
            color1 (str): Color for the bar plot. Default is 'blue'.
            color2 (str): Color for the radar plot. Default is 'green'.

        Returns:
            None

        Raises:
            None

        """
        plt.rcParams.update({'font.size': fontsize})  # Adjust font size
        
        if metrics is None:
            metrics = ['zcr_similarity', 'rhythm_similarity', 'chroma_similarity', 'energy_envelope_similarity', 'perceptual_similarity', 'stent_weighted_audio_similarity']

        if option == 'radar':
            num_metrics = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
            values = np.zeros(num_metrics)

            for i, metric in enumerate(metrics):
                if metric == 'zcr_similarity':
                    values[i] = self.zcr_similarity()
                elif metric == 'rhythm_similarity':
                    values[i] = self.rhythm_similarity()
                elif metric == 'chroma_similarity':
                    values[i] = self.chroma_similarity()
                elif metric == 'energy_envelope_similarity':
                    values[i] = self.energy_envelope_similarity()
                elif metric == 'perceptual_similarity':
                    values[i] = self.perceptual_similarity()
                elif metric == 'stent_weighted_audio_similarity':
                    values[i] = self.stent_weighted_audio_similarity()

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})  # Set projection to 'polar' for radar chart
            ax.plot(angles, values, color=color2, alpha=alpha)
            ax.fill(angles, values, color=color2, alpha=alpha)
            ax.set_xticks(angles)
            ax.set_xticklabels(metrics, fontsize=label_fontsize)  # Set default tick label fontsize
            ax.set_title(title, fontsize=title_fontsize)  # Set title fontsize

            # Show gridlines for the radar chart
            ax.grid(True)

            if savefig:
                plt.savefig('radar_plot.png', dpi=dpi)  # Save plot as an image file
            plt.show()

        elif option == 'bar':
            null_stream = open(os.devnull, 'w')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sys.stderr = null_stream
                metric_values = []
                for metric in metrics:
                    if metric == 'zcr_similarity':
                        metric_values.append(self.zcr_similarity())
                    elif metric == 'rhythm_similarity':
                        metric_values.append(self.rhythm_similarity())
                    elif metric == 'chroma_similarity':
                        metric_values.append(self.chroma_similarity())
                    elif metric == 'energy_envelope_similarity':
                        metric_values.append(self.energy_envelope_similarity())
                    elif metric == 'perceptual_similarity':
                        metric_values.append(self.perceptual_similarity())
                    elif metric == 'stent_weighted_audio_similarity':
                        metric_values.append(self.stent_weighted_audio_similarity())

                if len(metrics) != len(metric_values):
                    logging.info("Error: The number of metrics and metric values must be the same.")
                    return

                # Sort the metrics and values in descending order of values
                metric_values, metrics = zip(*sorted(zip(metric_values, metrics), reverse=True))

                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # Increase DPI
                ax.barh(metrics, metric_values, color=color1, alpha=alpha)
                ax.set_title(title, fontsize=title_fontsize)  # Set title fontsize

                ax.spines['top'].set_visible(False)  # Hide the top spine
                ax.spines['right'].set_visible(False)  # Hide the right spine

                # Hide gridlines for the bar chart
                ax.grid(False)

                if savefig:
                    plt.savefig('bar_plot.png', dpi=dpi)  # Save plot as an image file
                plt.show()

        elif option == 'all':
            num_metrics = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
            values = np.zeros(num_metrics)

            for i, metric in enumerate(metrics):
                if metric == 'zcr_similarity':
                    values[i] = self.zcr_similarity()
                elif metric == 'rhythm_similarity':
                    values[i] = self.rhythm_similarity()
                elif metric == 'chroma_similarity':
                    values[i] = self.chroma_similarity()
                elif metric == 'energy_envelope_similarity':
                    values[i] = self.energy_envelope_similarity()
                elif metric == 'perceptual_similarity':
                    values[i] = self.perceptual_similarity()
                elif metric == 'stent_weighted_audio_similarity':
                    values[i] = self.stent_weighted_audio_similarity()

            fig = plt.figure(figsize=figsize, dpi=dpi)

            # Bar plot
            ax1 = fig.add_subplot(1, 2, 1)
            metric_values = []
            for metric in metrics:
                if metric == 'zcr_similarity':
                    metric_values.append(self.zcr_similarity())
                elif metric == 'rhythm_similarity':
                    metric_values.append(self.rhythm_similarity())
                elif metric == 'chroma_similarity':
                    metric_values.append(self.chroma_similarity())
                elif metric == 'energy_envelope_similarity':
                    metric_values.append(self.energy_envelope_similarity())
                elif metric == 'perceptual_similarity':
                    metric_values.append(self.perceptual_similarity())
                elif metric == 'stent_weighted_audio_similarity':
                    metric_values.append(self.stent_weighted_audio_similarity())

            # Sort the metrics and values in descending order of values
            metric_values, metrics = zip(*sorted(zip(metric_values, metrics), reverse=True))

            ax1.barh(metrics, metric_values, color=color1, alpha=alpha)
            ax1.set_title(title, fontsize=title_fontsize)  # Set title fontsize

            ax1.spines['top'].set_visible(False)  # Hide the top spine
            ax1.spines['right'].set_visible(False)  # Hide the right spine

            # Hide gridlines for the bar chart
            ax1.grid(False)

            # Radar plot
            ax2 = fig.add_subplot(1, 2, 2, projection='polar')
            ax2.plot(angles, values, color=color2, alpha=alpha)
            ax2.fill(angles, values, color=color2, alpha=alpha)
            ax2.set_xticks(angles)
            ax2.set_xticklabels(metrics, fontsize=label_fontsize)  # Set default tick label fontsize
            ax2.set_title(title, fontsize=title_fontsize)  # Set title fontsize
            


            # Show gridlines for the radar chart
            ax2.grid(True)

            if savefig:
                plt.savefig('subplot_mosaic.png', dpi=dpi)  # Save plot as an image file
            plt.show()

        else:
            logging.info("Invalid plot option. Please choose 'radar', 'bar', or 'all'.")
