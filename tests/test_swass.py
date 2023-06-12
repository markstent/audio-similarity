import unittest
import os
import sys

# Get the parent directory of the current file
current_dir = os.getcwd()

# Add the parent directory to sys.path
sys.path.append(current_dir)


from audio_similarity.audio_similarity import AudioSimilarity

class TestAudioSimilarity(unittest.TestCase):

    def setUp(self):
        self.original_path = 'audio/original' # Replace with actual paths
        self.compare_path = 'audio/compare' # Replace with actual paths
        self.sample_rate = 22050
        self.sample_size = 1
        self.audio_similarity = AudioSimilarity(self.original_path, self.compare_path, self.sample_rate, sample_size=self.sample_size)

    def test_init(self):
        self.assertEqual(self.audio_similarity.original_path, self.original_path)
        self.assertEqual(self.audio_similarity.compare_path, self.compare_path)
        self.assertEqual(self.audio_similarity.sample_rate, self.sample_rate)
        self.assertEqual(self.audio_similarity.sample_size, self.sample_size)

    def test_parse_weights(self):
        weights_dict = self.audio_similarity.parse_weights([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
        expected_dict = {
            'zcr_similarity': 0.1,
            'rhythm_similarity': 0.2,
            'spectral_flux_similarity': 0.3,
            'energy_envelope_similarity': 0.1,
            'spectral_contrast_similarity': 0.1,
            'perceptual_similarity': 0.2
        }
        self.assertEqual(weights_dict, expected_dict)

    def test_load_audio_files(self):
        original_audios, compare_audios = self.audio_similarity.load_audio_files()
        self.assertTrue(len(original_audios) > 0)
        self.assertTrue(len(compare_audios) > 0)

    # We will only test one of the audio similarity calculation methods as they are quite similar. 
    # In a real testing scenario, we would test all of them.
    def test_zcr_similarity(self):
        similarity = self.audio_similarity.zcr_similarity()
        self.assertTrue(0 <= similarity <= 1)

    def test_stent_weighted_audio_similarity(self):
        similarity = self.audio_similarity.stent_weighted_audio_similarity(metrics='all')
        self.assertTrue(0 <= similarity['swass'] <= 1)

    def test_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            invalid_path = 'path/to/invalid/directory'
            AudioSimilarity(invalid_path, self.compare_path, self.sample_rate)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            AudioSimilarity(self.original_path, self.compare_path, self.sample_rate, weights=[0.1, 0.2])

if __name__ == '__main__':
    unittest.main()
