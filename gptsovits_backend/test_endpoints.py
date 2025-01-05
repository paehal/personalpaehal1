import requests
import base64
import json

def read_audio_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def test_zero_shot():
    print("Testing zero-shot TTS endpoint...")
    url = "http://localhost:8000/api/tts/zero-shot"
    audio = read_audio_file("test_audio_2s.wav")
    data = {
        "text": "Hello, this is a test.",
        "source_language": "en",
        "target_language": "ja",
        "reference_audio": audio,
        "mode": "zero-shot"
    }
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_few_shot():
    print("\nTesting few-shot TTS endpoint...")
    url = "http://localhost:8000/api/tts/few-shot"
    audio = read_audio_file("test_audio_3s.wav")
    data = {
        "text": "こんにちは、テストです。",
        "source_language": "ja",
        "target_language": "en",
        "training_audio": audio,
        "mode": "few-shot"
    }
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_invalid_input():
    print("\nTesting invalid input handling...")
    url = "http://localhost:8000/api/tts/zero-shot"
    data = {
        "text": "",  # Empty text should fail
        "source_language": "invalid",  # Invalid language
        "target_language": "ja",
        "reference_audio": "invalid_base64",  # Invalid base64
        "mode": "zero-shot"
    }
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_zero_shot()
    test_few_shot()
    test_invalid_input()
