"""GPT-SoVITS model implementation for Japanese TTS."""
from typing import Optional, Dict, Any, Union, cast
import logging
from functools import wraps
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import is_available
from torch.jit import inference_mode

logger = logging.getLogger(__name__)

# Define constants
DEVICE = "cuda" if is_available() else "cpu"

def inference_decorator(func):
    """Decorator for inference mode."""
    @wraps(func)
    @inference_mode
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class GPTSoVITSModel:
    """GPT-SoVITS model for Japanese text-to-speech synthesis."""
    
    def __init__(self, models: Dict[str, Any], device: str):
        """
        Initialize GPT-SoVITS model.
        
        Args:
            models: Dictionary containing loaded model components
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.models = models
        self.device = DEVICE
        self.s1 = models["s1"].to(device)
        self.s2_D = models["s2_D"].to(device)
        self.s2_G = models["s2_G"].to(device)
        self.uvr5 = models["uvr5"].to(device)
        self.asr = models["asr"]
        
        # Set models to evaluation mode
        self.s1.eval()
        self.s2_D.eval()
        self.s2_G.eval()
        self.uvr5.eval()
        
        logger.info("GPT-SoVITS model initialized successfully")
    
    @inference_decorator
    def _preprocess_audio(self, audio_path: Path) -> Optional[Any]:
        """Preprocess audio file for inference."""
        # Load and normalize audio
        y, sr = librosa.load(str(audio_path), sr=44100)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Apply voice separation using UVR5
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            
        # Convert to tensor and process
        audio_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        separated = self.uvr5(audio_tensor)
        vocals = separated['vocals'].cpu().numpy()[0]
        
        return torch.from_numpy(vocals.astype(np.float32)).to(self.device)
    
    @inference_decorator
    def _preprocess_text(self, text: str) -> Optional[Any]:
        """Preprocess Japanese text for inference."""
        # Use ASR model to get phoneme features
        segments, _ = self.asr.transcribe(text, language="ja")
        features = []
        for segment in segments:
            features.append(segment.tokens)
        features = np.concatenate(features)
        phoneme_features = torch.from_numpy(features.astype(np.float32)).to(self.device)
        return phoneme_features
    
    @inference_decorator
    def infer_zero_shot(
        self,
        text: str,
        reference_path: Path,
        output_path: Path,
    ) -> None:
        """
        Perform zero-shot inference.
        
        Args:
            text: Input Japanese text
            reference_path: Path to reference audio file
            output_path: Path to save synthesized audio
        """
        try:
            logger.info(f"Starting zero-shot inference for text: {text}")
            
            # Preprocess inputs
            ref_audio = self._preprocess_audio(reference_path)
            text_features = self._preprocess_text(text)
            
            # Stage 1: Text to hidden features
            hidden = self.s1(text_features, ref_audio)
            # Stage 2: Hidden features to waveform
            waveform = self.s2_G(hidden)
                
            # Post-process and save audio
            audio_numpy = waveform.cpu().numpy()
            sf.write(
                str(output_path),
                audio_numpy,
                44100,
                subtype='PCM_16'
            )
            
            logger.info(f"Zero-shot inference completed, saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Zero-shot inference failed: {str(e)}")
            raise RuntimeError(f"Zero-shot inference failed: {str(e)}")
            
    @inference_decorator
    def few_shot_train(
        self,
        training_path: Path,
    ) -> None:
        """
        Perform few-shot training on user's voice.
        
        Args:
            training_path: Path to training audio file (3-120 seconds)
        """
        try:
            logger.info("Starting few-shot training")
            
            # Load and validate training audio
            y, sr = librosa.load(str(training_path), sr=44100)
            duration = len(y) / sr
            
            if not (3 <= duration <= 120):
                raise ValueError("Training audio must be between 3 and 120 seconds")
                
            # Preprocess training audio
            audio_features = self._preprocess_audio(training_path)
            if audio_features is None:
                raise RuntimeError("Failed to preprocess training audio")
                
            # Perform few-shot training
            self.s1.train()
            self.s2_G.train()
            try:
                # Train stage 1 and 2 models
                hidden = self.s1.train_step(audio_features)
                self.s2_G.train_step(hidden, audio_features)
                
                logger.info("Few-shot training completed successfully")
            finally:
                # Ensure models are back in eval mode
                self.s1.eval()
                self.s2_G.eval()
                
        except Exception as e:
            logger.error(f"Few-shot training failed: {str(e)}")
            raise RuntimeError(f"Few-shot training failed: {str(e)}")
            
    @inference_decorator
    def infer_few_shot(
        self,
        text: str,
        output_path: Path,
    ) -> None:
        """
        Generate speech using few-shot trained model.
        
        Args:
            text: Input Japanese text
            output_path: Path to save synthesized audio
        """
        try:
            logger.info(f"Starting few-shot inference for text: {text}")
            
            # Preprocess text input
            text_features = self._preprocess_text(text)
            if text_features is None:
                raise RuntimeError("Failed to preprocess text")
                
            # Generate speech using trained model
            hidden = self.s1(text_features)
            waveform = self.s2_G(hidden)
            
            # Post-process and save audio
            audio_numpy = waveform.cpu().numpy()
            sf.write(
                str(output_path),
                audio_numpy,
                44100,
                subtype='PCM_16'
            )
            
            logger.info(f"Few-shot inference completed, saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Few-shot inference failed: {str(e)}")
            raise RuntimeError(f"Few-shot inference failed: {str(e)}")
