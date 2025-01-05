# GPT-SoVITS Integration Guide

[English]

This guide explains how to set up and use the GPT-SoVITS text-to-speech system in our backend service. The implementation focuses specifically on Japanese language support.

## Model Requirements

The following model files are required:

- Stage 1 Model: `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt`
- Stage 2 Models:
  - Discriminator: `s2D488k.pth`
  - Generator: `s2G488k.pth`
- UVR5 Model: Voice separation model (any .pth file in uvr5_weights directory)
- ASR Model: Faster Whisper model for Japanese speech recognition

## Environment Variables

Configure the following environment variables:

```env
MODEL_DIR=/path/to/models
TEMP_DIR=/path/to/temp
MAX_AUDIO_LENGTH=120  # Maximum audio length in seconds
```

## Usage Instructions

### Zero-Shot Inference

Zero-shot inference allows generating speech without training:

1. Input Requirements:
   - Japanese text
   - Reference audio (2-10 seconds)
2. API Endpoint: `POST /tts/zero-shot`
3. Response: Base64 encoded WAV audio

### Few-Shot Training and Inference

Few-shot adaptation allows training on a user's voice:

1. Training Requirements:
   - Training audio (3-120 seconds)
   - Japanese speech only
2. Inference Requirements:
   - Japanese text input
3. API Endpoints:
   - Training: `POST /tts/few-shot/train`
   - Inference: `POST /tts/few-shot/infer`
4. Response: Base64 encoded WAV audio

## GPU Requirements

- CUDA-compatible GPU recommended
- Minimum 8GB VRAM
- Falls back to CPU if GPU unavailable

---

[日本語]

このガイドでは、バックエンドサービスにおけるGPT-SoVITS音声合成システムのセットアップと使用方法について説明します。日本語音声のみをサポートしています。

## モデル要件

以下のモデルファイルが必要です：

- Stage 1モデル: `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt`
- Stage 2モデル:
  - 識別器: `s2D488k.pth`
  - 生成器: `s2G488k.pth`
- UVR5モデル: 音声分離モデル（uvr5_weightsディレクトリ内の.pthファイル）
- ASRモデル: 日本語音声認識用Faster Whisperモデル

## 環境変数

以下の環境変数を設定してください：

```env
MODEL_DIR=/path/to/models
TEMP_DIR=/path/to/temp
MAX_AUDIO_LENGTH=120  # 最大音声長（秒）
```

## 使用方法

### ゼロショット推論

学習なしで音声を生成できます：

1. 入力要件：
   - 日本語テキスト
   - 参照音声（2-10秒）
2. APIエンドポイント: `POST /tts/zero-shot`
3. レスポンス: Base64エンコードされたWAV音声

### フューショット学習と推論

ユーザーの声で学習できます：

1. 学習要件：
   - 学習用音声（3-120秒）
   - 日本語音声のみ
2. 推論要件：
   - 日本語テキスト入力
3. APIエンドポイント：
   - 学習: `POST /tts/few-shot/train`
   - 推論: `POST /tts/few-shot/infer`
4. レスポンス: Base64エンコードされたWAV音声

## GPU要件

- CUDA対応GPUを推奨
- 最小8GB VRAM
- GPU非対応の場合はCPUで動作
