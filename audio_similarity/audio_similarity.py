import numpy as np
import librosa
import pystoi
import os
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

class AudioSimilarity:
    """
    Calculates audio similarity metrics between original and generated audio files.

    Parameters:
        original_path (str): Path to the original audio file or directory containing multiple audio files.
        generated_path (str): Path to the generated audio file or directory containing multiple audio files.
        sample_rate (int): Sample rate of the audio files.
        weights (dict, optional): Weights assigned to each similarity metric. Defaults to None.

    Attributes:
        weights (dict): Dictionary of weights assigned to each similarity metric.
        sample_rate (int): Sample rate of the audio files.
        original_path (str): Path to the original audio file or directory.
        generated_path (str): Path to the generated audio file or directory.
        is_directory (list): List indicating if the original and generated paths are directories.
        original_audios (list): List of loaded original audio files.
        generated_audios (list): List of loaded generated audio files.

    Methods:
        calculate_similarity: Calculates the similarity score between original and generated audio files.
        zcr_similarity: Calculates the zero-crossing rate (ZCR) similarity score.
        rhythm_similarity: Calculates the rhythm similarity score.
        spectral_flux_similarity: Calculates the spectral flux similarity score.
        energy_envelope_similarity: Calculates the energy envelope similarity score.
        spectral_contrast_similarity: Calculates the spectral contrast similarity score.
        perceptual_similarity: Calculates the perceptual similarity score.
        parse_weights: Parses the weights specified by the user.

    """
    def __init__(self, original_path, generated_path, sample_rate, weights=None):
        """
        Initialize the AudioSimilarity object with the provided parameters.

        If weights are not specified, default weights are used for each similarity metric.

        Parameters:
            original_path (str): Path to the original audio file or directory containing multiple audio files.
            generated_path (str): Path to the generated audio file or directory containing multiple audio files.
            sample_rate (int): Sample rate of the audio files.
            weights (dict, optional): Weights assigned to each similarity metric. Defaults to None.

        """
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

        self.sample_rate = sample_rate
        self.original_path = original_path
        self.generated_path = generated_path

        # Check if the paths are directories or files
        self.is_directory = [os.path.isdir(path) for path in [original_path, generated_path]]

        # Load audio files once
        self.original_audios, self.generated_audios = self.load_audio_files()

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
        original_audios = []
        generated_audios = []
        valid_extensions = ('.mp3', '.flac', '.wav')

        if self.is_directory[0]:
            try:
                original_files = [os.path.join(self.original_path, f) for f in os.listdir(self.original_path) if f.endswith(valid_extensions)]
            except FileNotFoundError as e:
                print(f"Error loading original audio files: {e}")
                return [], []
        else:
            original_files = [self.original_path]

        if self.is_directory[1]:
            try:
                generated_files = [os.path.join(self.generated_path, f) for f in os.listdir(self.generated_path) if f.endswith(valid_extensions)]
            except FileNotFoundError as e:
                print(f"Error loading generated audio files: {e}")
                return [], []
        else:
            generated_files = [self.generated_path]

        for original_file in original_files:
            try:
                original_audio, _ = librosa.load(original_file, sr=self.sample_rate)
                original_audios.append(original_audio)
            except (FileNotFoundError, NoBackendError) as e:
                print(f"Error loading file {original_file}: {e}")
                continue
            except Exception as e:
                print(f"Error loading file {original_file}: {type(e).__name__}")
                continue

        for generated_file in generated_files:
            try:
                generated_audio, _ = librosa.load(generated_file, sr=self.sample_rate)
                generated_audios.append(generated_audio)
            except (FileNotFoundError, NoBackendError) as e:
                print(f"Error loading file {generated_file}: {e}")
                continue
            except Exception as e:
                print(f"Error loading file {generated_file}: {type(e).__name__}")
                continue

        return original_audios, generated_audios


    def zcr_similarity(self):
        """
        Calculate the Zero Crossing Rate (ZCR) similarity between the original audio and generated audio.

        Returns:
            float: The ZCR similarity value, ranging from 0 to 1.
                A higher value indicates a greater similarity between the ZCR of the original and generated audio.

        Raises:
            None

        Note:
            The ZCR similarity is calculated as the average similarity between the ZCR values of the original and generated audio.
            The ZCR represents the rate at which the audio waveform changes its sign (from positive to negative or vice versa).
            A higher ZCR indicates more rapid changes in the audio waveform, while a lower ZCR indicates smoother waveform.
            The ZCR similarity is calculated using the absolute difference between the ZCR values of the original and generated audio,
            normalized to a range of 0 to 1. A value of 1 indicates identical ZCR values, while a value of 0 indicates no similarity.

        """
        total_zcr_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_zcr = np.mean(np.abs(np.diff(np.sign(original_audio))) > 0)

            for generated_audio in self.generated_audios:
                generated_zcr = np.mean(np.abs(np.diff(np.sign(generated_audio))) > 0)
                zcr_similarity = 1 - np.abs(original_zcr - generated_zcr)
                total_zcr_similarity += zcr_similarity
                count += 1

        if count > 0:
            return total_zcr_similarity / count
        else:
            print("No original audio files loaded.")
            return None


    def rhythm_similarity(self):
        """
        Calculate the rhythm similarity between the original audio and generated audio.

        Returns:
            float: The rhythm similarity value, ranging from 0 to 1.
                A higher value indicates a greater similarity between the rhythm patterns of the original and generated audio.

        Raises:
            None

        Note:
            The rhythm similarity is calculated based on the onset patterns of the original and generated audio.
            Onsets are the points in time where significant events occur in the audio, such as beats or musical events.
            The onset vectors are constructed for both the original and generated audio, indicating the presence (1) or absence (0) of onsets.
            The rhythm similarity is calculated using the correlation coefficient between the original and generated onset vectors,
            normalized to a range of 0 to 1. A value of 1 indicates identical rhythm patterns, while a value of 0 indicates no similarity.

        """
        total_rhythm_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            min_length = len(original_audio)
            original_onset_vector = np.zeros(min_length)
            original_onsets = librosa.onset.onset_detect(y=original_audio, sr=self.sample_rate, units='time')
            original_onsets = np.array(original_onsets) * self.sample_rate
            original_onsets = original_onsets[original_onsets < min_length]
            original_onset_vector[original_onsets.astype(int)] = 1

            for generated_audio in self.generated_audios:
                min_length = min(min_length, len(generated_audio))
                generated_onset_vector = np.zeros(min_length)
                generated_onsets = librosa.onset.onset_detect(y=generated_audio[:min_length], sr=self.sample_rate, units='time')
                generated_onsets = np.array(generated_onsets) * self.sample_rate
                generated_onsets = generated_onsets[generated_onsets < min_length]
                generated_onset_vector[generated_onsets.astype(int)] = 1

                rhythm_similarity = (np.corrcoef(original_onset_vector[:min_length], generated_onset_vector[:min_length])[0, 1] + 1) / 2
                total_rhythm_similarity += rhythm_similarity
                count += 1

        if count > 0:
            return total_rhythm_similarity / count
        else:
            print("No original audio files loaded.")
            return None


    def chroma_similarity(self):
        """
        Calculate the chroma similarity between the original audio and generated audio.

        Returns:
            float: The chroma similarity value, ranging from 0 to 1.
                A higher value indicates a greater similarity between the chroma patterns of the original and generated audio.

        Raises:
            None

        Note:
            The chroma similarity is calculated based on the chroma features of the original and generated audio.
            Chroma features represent the distribution of pitch classes in an audio signal, providing information about the tonal content.
            The chroma features are computed using the Constant-Q Transform (CQT).
            The chroma similarity is calculated as the average absolute difference between the original and generated chroma matrices,
            normalized to a range of 0 to 1. A value of 1 indicates identical chroma patterns, while a value of 0 indicates no similarity.

        """
        total_chroma_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            for generated_audio in self.generated_audios:
                original_chroma = librosa.feature.chroma_cqt(y=original_audio, sr=self.sample_rate)
                generated_chroma = librosa.feature.chroma_cqt(y=generated_audio, sr=self.sample_rate)

                min_length = min(original_chroma.shape[1], generated_chroma.shape[1])
                original_chroma = original_chroma[:, :min_length]
                generated_chroma = generated_chroma[:, :min_length]

                chroma_similarity = 1 - np.mean(np.abs(original_chroma - generated_chroma))
                total_chroma_similarity += chroma_similarity
                count += 1

        if count > 0:
            return total_chroma_similarity / count
        else:
            print("No original audio files loaded.")
            return None



    def energy_envelope_similarity(self):
        """
        Calculate the energy envelope similarity between the original audio and generated audio.

        Returns:
            float: The energy envelope similarity value, ranging from 0 to 1.
                A higher value indicates a greater similarity between the energy envelopes of the original and generated audio.

        Raises:
            None

        Note:
            The energy envelope similarity is calculated based on the energy envelopes of the original and generated audio signals.
            The energy envelope represents the temporal variation of the signal's energy.
            The energy envelope is computed by taking the absolute value of the audio signal.
            The energy envelope similarity is calculated as the normalized correlation coefficient between the original and generated energy envelopes.
            The correlation coefficient ranges from -1 to 1, and the energy envelope similarity is obtained by normalizing the coefficient to a range of 0 to 1.
            A value of 1 indicates identical energy envelope patterns, while a value of 0 indicates no similarity.

        """
        total_energy_envelope_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            min_length = len(original_audio)
            original_energy_envelope = np.abs(original_audio[:min_length])

            for generated_audio in self.generated_audios:
                min_length = min(min_length, len(generated_audio))
                generated_energy_envelope = np.abs(generated_audio[:min_length])

                energy_envelope_similarity = (np.corrcoef(original_energy_envelope[:min_length], generated_energy_envelope[:min_length])[0, 1] + 1) / 2
                total_energy_envelope_similarity += energy_envelope_similarity
                count += 1

        if count > 0:
            return total_energy_envelope_similarity / count
        else:
            print("No original audio files loaded.")
            return None


    def spectral_contrast_similarity(self):
        """
        Calculate the spectral contrast similarity between the original audio and generated audio.

        Returns:
            float: The spectral contrast similarity value, ranging from 0 to 1.
                A higher value indicates a greater similarity between the spectral contrast patterns of the original and generated audio.

        Raises:
            None

        Note:
            The spectral contrast similarity is calculated based on the spectral contrast features of the original and generated audio signals.
            The spectral contrast feature measures the difference in magnitude between peaks and valleys in different frequency bands.
            The spectral contrast similarity is calculated as the mean absolute difference between the spectral contrast features of the original and generated audio.
            The difference is normalized by dividing it by the maximum value between the absolute values of the original and generated contrast features.
            The resulting similarity value ranges from 0 to 1, where a higher value indicates a greater similarity between the spectral contrast patterns.
            A value of 1 indicates identical spectral contrast patterns, while a value of 0 indicates no similarity.

        """
        if not self.original_audios or not self.generated_audios:
            print("No audio files loaded.")
            return None

        total_spectral_contrast_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_contrast = librosa.feature.spectral_contrast(y=original_audio, sr=self.sample_rate)
            min_columns = original_contrast.shape[1]

            for generated_audio in self.generated_audios:
                generated_contrast = librosa.feature.spectral_contrast(y=generated_audio, sr=self.sample_rate)
                min_columns = min(min_columns, generated_contrast.shape[1])

            original_contrast = original_contrast[:, :min_columns]

            for generated_audio in self.generated_audios:
                generated_contrast = librosa.feature.spectral_contrast(y=generated_audio, sr=self.sample_rate)
                generated_contrast = generated_contrast[:, :min_columns]
                contrast_similarity = np.mean(np.abs(original_contrast - generated_contrast))
                normalized_similarity = 1 - contrast_similarity / np.max([np.abs(original_contrast), np.abs(generated_contrast)])
                total_spectral_contrast_similarity += normalized_similarity
                count += 1

        if count > 0:
            return total_spectral_contrast_similarity / count
        else:
            print("No original audio files loaded.")
            return None


    def perceptual_similarity(self, sr=16000):
        """
        Calculate the perceptual similarity between the original audio and generated audio.

        Args:
            sr (int): The target sample rate for resampling the audio signals. Default is 16000.

        Returns:
            float: The perceptual similarity value, ranging from 0 to 1.
                A higher value indicates a greater perceptual similarity between the original and generated audio.

        Raises:
            None

        Note:
            The perceptual similarity is calculated using the Short-Time Objective Intelligibility (STOI) metric.
            The STOI measures the similarity of the intelligibility of two speech signals, which corresponds to the perceived similarity by human listeners.
            The original and generated audio signals are resampled to the specified sample rate before calculating the STOI score.
            The STOI score is normalized to the range of 0 to 1, where a higher value indicates a greater perceptual similarity.
            A value of 1 indicates perfect perceptual similarity, while a value of 0 indicates no perceptual similarity.

        """
        if not self.original_audios or not self.generated_audios:
            print("No audio files loaded.")
            return None

        total_perceptual_similarity = 0
        count = 0

        for i, original_audio in enumerate(self.original_audios):
            for j, generated_audio in enumerate(self.generated_audios):
                min_length = min(len(original_audio), len(generated_audio))
                array1_16k = librosa.resample(y=original_audio[:min_length], orig_sr=self.sample_rate, target_sr=sr)
                array2_16k = librosa.resample(y=generated_audio[:min_length], orig_sr=self.sample_rate, target_sr=sr)
                score = pystoi.stoi(array1_16k, array2_16k, sr)
                score_normalized = (score + 1) / 2
                total_perceptual_similarity += score_normalized
                count += 1

        if count > 0:
            return total_perceptual_similarity / count
        else:
            print("No original audio files loaded.")
            return None


    def stent_weighted_audio_similarity(self):
        """
        Calculate the overall weighted audio similarity between the original audio and generated audio.

        Returns:
            float: The overall weighted audio similarity value, ranging from 0 to 1.
                A higher value indicates a greater overall similarity between the original and generated audio.

        Raises:
            None

        Note:
            The overall weighted audio similarity is calculated by combining multiple individual audio similarity metrics.
            The individual metrics used are:
            - Zero Crossing Rate (ZCR) similarity
            - Rhythm similarity
            - Chroma similarity
            - Energy envelope similarity
            - Spectral contrast similarity
            - Perceptual similarity

            Each metric is calculated for all combinations of the original and generated audio signals.
            The individual metric values are then multiplied by their respective weights, which can be customized.
            The weighted metric values are summed to obtain the overall similarity score.

            The weights for the individual metrics can be set using the `weights` parameter in the `AudioSimilarity` constructor.
            By default, equal weights are assigned to all metrics.

            The overall similarity value is normalized to the range of 0 to 1, where a higher value indicates a greater overall similarity.
            A value of 1 indicates perfect overall similarity, while a value of 0 indicates no overall similarity.

            If no audio files are loaded, the method returns `None` and displays an error message.
        """
        if not self.original_audios or not self.generated_audios:
            print("No audio files loaded.")
            return None

        num_original_audios = len(self.original_audios)
        num_generated_audios = len(self.generated_audios)

        zcr_similarities = np.zeros((num_original_audios, num_generated_audios))
        rhythm_similarities = np.zeros((num_original_audios, num_generated_audios))
        chroma_similarity = np.zeros((num_original_audios, num_generated_audios))
        energy_envelope_similarities = np.zeros((num_original_audios, num_generated_audios))
        spectral_contrast_similarities = np.zeros((num_original_audios, num_generated_audios))
        perceptual_similarities = np.zeros((num_original_audios, num_generated_audios))

        for i, original_audio in enumerate(self.original_audios):
            for j, generated_audio in enumerate(self.generated_audios):
                self.original_audio = original_audio
                self.generated_audio = generated_audio
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
        count = num_original_audios * num_generated_audios

        if count > 0:
            return total_similarity / count
        else:
            print("No audio files loaded.")
            return None
        

    def plot(self, metrics, option='spider', figsize=(8, 6), color='blue', alpha=0.5, title=None):
        """
        Create a spider plot or horizontal bar plot for the specified metrics.

        Parameters:
            metrics (list): List of metrics to plot. These metrics represent the audio similarity measures that will be visualized.
            option (str): Option to specify the plot type. 'all' for spider plot, 'bar' for horizontal bar plot.
            figsize (tuple): Figure size (width, height) in inches.
            color (str or list): Color(s) of the plot. Specify a single color or a list of colors for each metric.
            alpha (float or list): Transparency level(s) of the plot. Specify a single alpha value or a list of alpha values for each metric.
            title (str): Title of the plot.

        Returns:
            None

        Explanation:
        - The `metrics` parameter expects a list of strings, where each string represents an audio similarity metric that you want to include in the plot.
        - The `option` parameter allows you to choose between two plot types: 'all' for a spider plot and 'bar' for a horizontal bar plot.
        - The `figsize` parameter allows you to specify the size of the figure in inches.
        - The `color` parameter allows you to specify the color(s) of the plot. You can provide a single color as a string, or a list of colors corresponding to each metric.
        - The `alpha` parameter allows you to specify the transparency level(s) of the plot. You can provide a single alpha value as a float, or a list of alpha values corresponding to each metric.
        - The `title` parameter allows you to set a title for the plot.
        - If `option` is set to 'all', a spider plot will be created, where each metric is represented by a point on the plot. The values of the metrics will be plotted as radial lines extending from the center of the plot.
        - If `option` is set to 'bar', a horizontal bar plot will be created, where each metric is represented by a horizontal bar. The length of each bar corresponds to the value of the metric.
        - The plot will be displayed using `plt.show()`.

        Usage examples:
        1. Spider plot:
            audio_similarity.plot(metrics=['zcr_similarity', 'rhythm_similarity', 'chroma_similarity'], option='all', figsize=(8, 6), color='blue', alpha=0.5, title='Audio Similarity')
        2. Horizontal bar plot:
            audio_similarity.plot(metrics=['zcr_similarity', 'rhythm_similarity', 'chroma_similarity'], option='bar', figsize=(8, 6), color=['red', 'green', 'blue'], alpha=[0.5, 0.6, 0.7], title='Audio Similarity Metrics')
        """
        if option == 'spider':
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
                elif metric == 'mfcc_similarity':
                    values[i] = self.perceptual_similarity()
                elif metric == 'stent_weighted_audio_similarity':
                    values[i] = self.stent_weighted_audio_similarity()

            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
            ax.plot(angles, values, color=color, alpha=alpha)
            ax.fill(angles, values, color=color, alpha=alpha)
            ax.set_xticks(angles)
            ax.set_xticklabels(metrics)
            ax.set_title(title)

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
                    elif metric == 'mfcc_similarity':
                        metric_values.append(self.perceptual_similarity())
                    elif metric == 'stent_weighted_audio_similarity':
                        metric_values.append(self.stent_weighted_audio_similarity())

                if len(metrics) != len(metric_values):
                    print("Error: The number of metrics and metric values must be the same.")
                    return
                
                # Sort the metrics and values in descending order of values
                metric_values, metrics = zip(*sorted(zip(metric_values, metrics), reverse=True))

                fig, ax = plt.subplots(figsize=figsize)
                ax.barh(metrics, metric_values, color=color, alpha=alpha)
                ax.set_title(title)

                plt.show(block=True)
            # Restore stderr
            sys.stderr = sys.__stderr__

        else:
            print("Invalid plot option. Please choose 'spider' or 'bar'.")

