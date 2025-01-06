"""Test cases for Japanese text processing functionality.

日本語テキスト処理機能のテストケース
"""
from app.models.text_utils import JapaneseTextProcessor


def test_text_normalization():
    """Test Japanese text normalization.

    日本語テキスト正規化のテスト
    """
    processor = JapaneseTextProcessor()

    # Test basic punctuation normalization
    # 基本的な句読点の正規化テスト
    assert processor.normalize_text("こんにちは！！") == "こんにちは!"
    assert processor.normalize_text("おはよう。。。") == "おはよう."
    assert processor.normalize_text("さようなら、、、") == "さようなら,"

    # Test multiple punctuation marks
    # 複数の句読点のテスト
    assert processor.normalize_text("こんにちは！？") == "こんにちは!?"
    assert processor.normalize_text("本当ですか？！") == "本当ですか?!"

    # Test symbol conversion
    # 記号変換のテスト
    assert processor.normalize_text("50％") == "50パーセント"
    assert processor.normalize_text("テスト：テスト") == "テスト,テスト"


def test_phoneme_conversion_basic():
    """Test basic phoneme conversion without prosody.

    プロソディなしの基本的な音素変換テスト
    """
    processor = JapaneseTextProcessor()

    # Test basic conversion
    # 基本的な変換テスト
    phonemes = processor.text_to_phonemes("こんにちは", with_prosody=False)
    assert "k" in phonemes
    assert "o" in phonemes
    assert "n" in phonemes
    assert "ch" in phonemes
    assert "i" in phonemes
    assert "h" in phonemes
    assert "a" in phonemes


def test_phoneme_conversion_with_prosody():
    """Test phoneme conversion with prosody markers.

    プロソディマーカー付きの音素変換テスト
    """
    processor = JapaneseTextProcessor()

    # Test start/end markers
    # 開始/終了マーカーのテスト
    phonemes = processor.text_to_phonemes("こんにちは")
    assert phonemes[0] == "^"  # Start marker / 開始マーカー
    assert phonemes[-1] == "$"  # End marker / 終了マーカー

    # Test question form
    # 疑問形のテスト
    phonemes = processor.text_to_phonemes("こんにちは？")
    assert phonemes[-1] == "?"  # Question marker / 疑問符マーカー


def test_mixed_text_handling():
    """Test handling of mixed Japanese and punctuation.

    日本語と句読点の混合テキスト処理テスト
    """
    processor = JapaneseTextProcessor()

    # Test mixed text
    # 混合テキストのテスト
    text = "こんにちは！Hello！さようなら。"
    phonemes = processor.text_to_phonemes(text)

    # Verify basic structure
    # 基本構造の確認
    assert phonemes[0] == "^"  # Start marker / 開始マーカー
    assert "!" in phonemes  # Exclamation mark / 感嘆符
    assert "." in phonemes  # Period / 句点
    assert phonemes[-1] == "$"  # End marker / 終了マーカー


def test_error_handling():
    """Test error handling in text processing.

    テキスト処理のエラーハンドリングテスト
    """
    processor = JapaneseTextProcessor()

    # Test empty string
    # 空文字列のテスト
    assert processor.normalize_text("") == ""
    assert processor.text_to_phonemes("") == []

    # Test whitespace
    # 空白文字のテスト
    assert processor.normalize_text("  ") == "  "
    assert processor.text_to_phonemes("  ") == []
