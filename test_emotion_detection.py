"""
Unit tests for Emotion Detection application.
Mocks Watson NLP API responses for deterministic offline test verification.
"""
import json
import unittest
from unittest.mock import patch, MagicMock
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test suite for emotion detection function."""

    def setUp(self):
        """Set up mock for requests.post to simulate Watson NLP responses."""
        self.mock_patcher = patch('requests.post')
        self.mock_post = self.mock_patcher.start()

        def mock_side_effect(*args, **kwargs):
            del args
            payload = kwargs.get('json', {})
            text = payload.get('raw_document', {}).get('text', '')

            mock_resp = MagicMock()
            if not text or text.strip() == "":
                mock_resp.status_code = 400
                mock_resp.text = json.dumps({"code": 400, "message": "Invalid text input."})
                return mock_resp

            emotion_map = {
                "I am glad this happened": {
                    'anger': 0.01, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.95, 'sadness': 0.02
                },
                "I am really mad about this": {
                    'anger': 0.95, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.02
                },
                "I feel disgusted just hearing about this": {
                    'anger': 0.01, 'disgust': 0.95, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.02
                },
                "I am so sad about this": {
                    'anger': 0.01, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.95
                },
                "I am really afraid that this will happen": {
                    'anger': 0.01, 'disgust': 0.01, 'fear': 0.95, 'joy': 0.01, 'sadness': 0.02
                },
            }

            emotion = emotion_map.get(text, {
                'anger': 0.2, 'disgust': 0.2, 'fear': 0.2, 'joy': 0.2, 'sadness': 0.2
            })
            mock_resp.status_code = 200
            mock_resp.text = json.dumps({
                "emotionPredictions": [{"emotion": emotion}]
            })
            return mock_resp

        self.mock_post.side_effect = mock_side_effect

    def tearDown(self):
        """Stop mock patcher."""
        self.mock_patcher.stop()

    def test_joy(self):
        """Test dominant emotion detection for joy."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result['dominant_emotion'], 'joy')
        self.assertGreater(result['joy'], result['anger'])

    def test_anger(self):
        """Test dominant emotion detection for anger."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result['dominant_emotion'], 'anger')
        self.assertGreater(result['anger'], result['joy'])

    def test_disgust(self):
        """Test dominant emotion detection for disgust."""
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result['dominant_emotion'], 'disgust')
        self.assertGreater(result['disgust'], result['joy'])

    def test_sadness(self):
        """Test dominant emotion detection for sadness."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result['dominant_emotion'], 'sadness')
        self.assertGreater(result['sadness'], result['joy'])

    def test_fear(self):
        """Test dominant emotion detection for fear."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result['dominant_emotion'], 'fear')
        self.assertGreater(result['fear'], result['joy'])

    def test_error_handling_status_400(self):
        """Test handling of HTTP 400 / blank input returning None for all emotion fields."""
        result = emotion_detector("")
        expected = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
