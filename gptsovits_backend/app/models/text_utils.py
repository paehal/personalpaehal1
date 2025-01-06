"""Japanese text processing utilities for GPT-SoVITS.

日本語テキスト処理ユーティリティ for GPT-SoVITS
"""
import re
from typing import List
import pyopenjtalk

class JapaneseTextProcessor:
    """Japanese text processor for GPT-SoVITS.
    
    GPT-SoVITS用の日本語テキストプロセッサー
    """
    def __init__(self):
        self._setup_regex_patterns()
        
    def _setup_regex_patterns(self):
        """Set up regex patterns for text processing.
        
        テキスト処理用の正規表現パターンを設定
        """
        # Regular expression matching Japanese without punctuation marks
        self._japanese_characters = re.compile(
            r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )
        
        # Regular expression matching non-Japanese characters or punctuation marks
        self._japanese_marks = re.compile(
            r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )
        
        # Punctuation replacement mapping
        self._punctuation_map = {
            "：": ",",
            "；": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": ".",
            "·": ",",
            "、": ",",
            "...": "…",
        }
        
        # Symbol to Japanese mapping
        self._symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]
        
    def normalize_text(self, text: str) -> str:
        """Normalize Japanese text by standardizing punctuation and symbols.
        
        句読点や記号を標準化して日本語テキストを正規化します。

        Args:
            text (str): Input text / 入力テキスト

        Returns:
            str: Normalized text / 正規化されたテキスト
        """
        # Convert symbols to Japanese words
        for regex, replacement in self._symbols_to_japanese:
            text = re.sub(regex, replacement, text)
            
        # Replace punctuation marks
        for old, new in self._punctuation_map.items():
            text = text.replace(old, new)
            
        # Remove consecutive punctuation marks
        text = re.sub(r'([,.!?])[,.!?]+', r'\1', text)
        
        return text
        
    def text_to_phonemes(self, text: str, with_prosody: bool = True) -> List[str]:
        """Convert Japanese text to phoneme sequence.
        
        日本語テキストを音素列に変換します。

        Args:
            text (str): Input text / 入力テキスト
            with_prosody (bool): Include prosody information / プロソディ情報を含めるかどうか

        Returns:
            List[str]: List of phonemes / 音素のリスト
        """
        text = self.normalize_text(text)
        text = text.lower()  # Convert English to lowercase
        
        sentences = re.split(self._japanese_marks, text)
        marks = re.findall(self._japanese_marks, text)
        
        phonemes = []
        for i, sentence in enumerate(sentences):
            if re.match(self._japanese_characters, sentence):
                if with_prosody:
                    # Use OpenJTalk with prosody
                    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(sentence))
                    phones = self._extract_phonemes_with_prosody(labels)
                    phonemes.extend(phones)
                else:
                    # Use basic OpenJTalk phoneme conversion
                    p = pyopenjtalk.g2p(sentence)
                    phonemes.extend(p.split(" "))
            
            # Add punctuation marks
            if i < len(marks):
                if marks[i] != " ":  # Skip spaces to prevent UNK tokens
                    phonemes.append(marks[i].replace(" ", ""))
                    
        return phonemes
        
    def _extract_phonemes_with_prosody(self, labels: List[str]) -> List[str]:
        """Extract phonemes with prosody information from OpenJTalk labels.
        
        OpenJTalkラベルから韻律情報付きの音素を抽出します。

        Args: 
            labels (List[str]): OpenJTalk full-context labels
            
        Returns:
            List[str]: Phonemes with prosody markers
        """
        phones = []
        N = len(labels)
        
        
        for n in range(N):
            lab_curr = labels[n]
            
            # Extract current phoneme
            p3 = re.search(r"\-(.*?)\+", lab_curr)
            if not p3:  # Skip if no phoneme found
                continue
            p3 = p3.group(1)
            
            # Handle silence and pauses
            if p3 == "sil":
                if n == 0:
                    phones.append("^")  # Start marker
                elif n == N - 1:
                    # Check question form
                    e3 = re.search(r"!(\d+)_", lab_curr)
                    phones.append("?" if e3 and e3.group(1) == "1" else "$")
                continue
            elif p3 == "pau":
                phones.append("_")
                continue
                
            phones.append(p3)
            
            # Add prosody markers if we have a next label
            if n + 1 < N:
                lab_next = labels[n + 1]
                
                # Extract accent features
                a1 = re.search(r"/A:([0-9\-]+)\+", lab_curr)
                a2 = re.search(r"\+(\d+)\+", lab_curr)
                a3 = re.search(r"\+(\d+)/", lab_curr)
                f1 = re.search(r"/F:(\d+)_", lab_curr)
                a2_next = re.search(r"\+(\d+)\+", lab_next)
                
                # Convert to integers, defaulting to 0 if not found
                a1 = int(a1.group(1)) if a1 else 0
                a2 = int(a2.group(1)) if a2 else 0
                a3 = int(a3.group(1)) if a3 else 0
                f1 = int(f1.group(1)) if f1 else 0
                a2_next = int(a2_next.group(1)) if a2_next else 0
                
                # Add prosody markers
                if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                    phones.append("#")  # Accent phrase border
                elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                    phones.append("]")  # Pitch falling
                elif a2 == 1 and a2_next == 2:
                    phones.append("[")  # Pitch rising
                    
        return phones
