# テストデータディレクトリ / Test Data Directory

このディレクトリには、GPT-SoVITSの動作検証に使用するテストデータが含まれています。
This directory contains test data used for verifying GPT-SoVITS functionality.

## ディレクトリ構造 / Directory Structure

```
test_data/
├── test_texts.json    # テストテキストデータ / Test text data
├── reference_audio/   # Few-shot TTS用の参照音声 / Reference audio for Few-shot TTS
└── outputs/          # テスト出力の保存先 / Test output storage
```

## テストデータの説明 / Test Data Description

### テストテキスト / Test Texts
- 基本的なテストケース / Basic test cases
  - 短い文章での基本機能テスト / Basic functionality tests with short sentences
  - 一般的な会話表現 / Common conversational expressions

- 複雑なテストケース / Complex test cases
  - 自然な抑揚のテスト / Natural intonation tests
  - 感情表現のテスト / Emotional expression tests
  - 長文でのテスト / Long text tests

### 参照音声 / Reference Audio
- Few-shot TTS用の短い日本語音声サンプル / Short Japanese voice samples for Few-shot TTS
  - 3-5秒の短い発話 / 3-5 second short utterances
  - クリアな録音品質 / Clear recording quality
  - 様々な話者の音声 / Voice samples from different speakers

## 使用方法 / Usage

1. Zero-shot TTSテスト / Zero-shot TTS Testing
   ```python
   # test_texts.jsonからテキストを読み込む例 / Example of loading text from test_texts.json
   with open('test_data/test_texts.json') as f:
       test_data = json.load(f)
   zero_shot_texts = test_data['zero_shot']
   ```

2. Few-shot TTSテスト / Few-shot TTS Testing
   ```python
   # Few-shot用の参照音声を使用する例 / Example of using reference audio for Few-shot
   reference_audio = 'test_data/reference_audio/sample_001.wav'
   few_shot_texts = test_data['few_shot']
   ```

## 注意事項 / Notes

- すべてのテストデータは日本語に特化しています
- 音声ファイルは24kHzのサンプリングレートを使用
- テスト結果は`outputs`ディレクトリに保存されます
- All test data is specialized for Japanese
- Audio files use 24kHz sampling rate
- Test results are saved in the `outputs` directory

## 参照音声の要件 / Reference Audio Requirements

1. 録音品質 / Recording Quality
   - サンプリングレート: 24kHz / Sampling rate: 24kHz
   - 無響室または低ノイズ環境での録音 / Recorded in anechoic or low-noise environment

2. 音声の長さ / Audio Length
   - 推奨: 3-5秒 / Recommended: 3-5 seconds
   - 最大: 10秒 / Maximum: 10 seconds

3. 発話内容 / Speech Content
   - クリアな発音の日本語 / Clear Japanese pronunciation
   - 自然な抑揚を含む / Including natural intonation
