# GPT-SoVITS Integration Documentation
# GPT-SoVITS 統合ドキュメント

## テキスト処理 / Text Processing

日本語テキスト処理の実装について説明します。
This section describes the Japanese text processing implementation.

### pyopenjtalkを使用した音素変換 / Phoneme Conversion Using pyopenjtalk

pyopenjtalkを使用して日本語テキストを音素シーケンスに変換します：
Converting Japanese text to phoneme sequences using pyopenjtalk:

- 基本的な音素抽出 / Basic phoneme extraction
- 「は」の音の特別な処理 / Special handling for は sound
- 開始・終了マーカー（^、$）/ Start and end markers (^, $)
- 疑問符の保持 / Question mark preservation

例 / Example:
```python
入力 / Input: こんにちは！
出力 / Output: ['^', 'k', 'o', 'N', 'n', 'i', 'ch', 'i', 'h', 'a', '!', '$']
```

### プロソディマーカーのサポート / Support for Prosody Markers

自然な発話リズムのためのプロソディ情報：
Prosody information for natural speech rhythm:

- アクセント句境界（#）/ Accent phrase borders (#)
- ピッチ上昇（[）/ Pitch rising ([)
- ピッチ下降（]）/ Pitch falling (])
- 休止（_）/ Pauses (_)

例 / Example:
```python
入力 / Input: 今日は晴れです。
出力 / Output: ['^', 'k', 'y', 'o', '[', 'u', '#', 'h', 'a', ']', '_', 'h', 'a', 'r', 'e', '_', 'd', 'e', 's', 'u', '.', '$']
```

### テキスト正規化 / Text Normalization

テキスト正規化の処理内容：
Text normalization process includes:

1. 句読点の標準化 / Punctuation standardization
   - 日本語からASCIIへの変換 / Japanese to ASCII conversion
   - 例：「。」→「.」/ Example: 「。」→ "."

2. 記号変換 / Symbol conversion
   - パーセント記号など / Percentage signs and others
   - 例：「％」→「パーセント」/ Example: "%" → "percent"

3. 特殊なケース / Special cases
   - 複数の句読点の処理 / Multiple punctuation handling
   - スペースの正規化 / Space normalization

## エラー処理 / Error Handling

システムは以下のエラーケースに対応：
The system handles the following error cases:

- 空の入力 / Empty input
- 無効な文字 / Invalid characters
- 混合テキスト / Mixed text

## テスト / Testing

実装には以下のテストが含まれています：
The implementation includes the following tests:

```python
- テキスト正規化テスト / Text normalization tests
- 基本的な音素変換テスト / Basic phoneme conversion tests
- プロソディマーカー処理テスト / Prosody marker handling tests
- 混合テキストテスト / Mixed text tests
- エラーケーステスト / Error case tests
```

## 依存関係 / Dependencies

必要なパッケージ：
Required packages:

- pyopenjtalk
- re (Python標準ライブラリ / Python standard library)
