import warnings

from audio_similarity.audio_similarity import AudioSimilarity

warnings.filterwarnings("ignore", category=UserWarning)

# Example usage: Calculate zcr_similarity
original_audio = '/Users/mark.stent/Dropbox/Data_Science/Python/audio_similarity/audio/original'  # Load the original audio data as a numpy array
compare_audio = '/Users/mark.stent/Dropbox/Data_Science/Python/audio_similarity/audio/generated'  # Load the generated audio data as a numpy array
sr = 44100

# Create an instance of AudioSimilarity with custom weights in dictionary (optional)
weights_dict = {
    'zcr_similarity': 0.3,
    'rhythm_similarity': 0.2,
    'spectral_flux_similarity': 0.15,
    'energy_envelope_similarity': 0.15,
    'spectral_contrast_similarity': 0.1,
    'perceptual_similarity': 0.1
}
# Create an instance of AudioSimilarity with custom weights in list (optional)
weights_list = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]

audio_similarity = AudioSimilarity(original_audio, compare_audio, sr, weights_dict)

# Calculate and print all the metrics
metrics = {
    #'zcr_similarity': audio_similarity.zcr_similarity(),
    #'rhythm_similarity': audio_similarity.rhythm_similarity(),
    #'chroma_similarity': audio_similarity.chroma_similarity(),
    #'energy_envelope_similarity': audio_similarity.energy_envelope_similarity(),
    #'perceptual_similarity': audio_similarity.perceptual_similarity(),
    #'spectral_contrast_similarity': audio_similarity.spectral_contrast_similarity(),
    'stent_weighted_audio_similarity': audio_similarity.stent_weighted_audio_similarity()
}

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")
    
# Plot with a spider or bar chart
    
#audio_similarity.plot(metrics=['zcr_similarity', 'rhythm_similarity', 'chroma_similarity', 'energy_envelope_similarity'],option='bar',figsize=(8, 6),color='red',alpha=0.5, title='Audio Similarity Metrics')



