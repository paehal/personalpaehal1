import librosa
import os
import soundfile as sf

def verify_audio(file_path, min_len, max_len, purpose):
    """Verify audio file exists and meets length requirements."""
    print(f"\nVerifying {purpose} audio file...")
    
    if not os.path.exists(file_path):
        print(f"ERROR: {purpose} file not found at {file_path}")
        return False
    
    try:
        # First try with soundfile which is faster
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
            print(f"{purpose} audio duration: {duration:.2f}s")
            
            if duration < min_len or duration > max_len:
                print(f"ERROR: {purpose} audio length must be between {min_len}-{max_len}s")
                return False
            
            print(f"✓ {purpose} audio file verified successfully")
            return True
            
    except Exception as e:
        print(f"WARNING: SoundFile failed, trying librosa: {str(e)}")
        try:
            # Fallback to librosa if soundfile fails
            y, sr = librosa.load(file_path)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"{purpose} audio duration: {duration:.2f}s")
            
            if duration < min_len or duration > max_len:
                print(f"ERROR: {purpose} audio length must be between {min_len}-{max_len}s")
                return False
                
            print(f"✓ {purpose} audio file verified successfully")
            return True
            
        except Exception as e:
            print(f"ERROR loading {purpose} audio: {str(e)}")
            return False

def main():
    print("Starting audio file verification...")
    
    # Verify reference audio (zero-shot)
    ref_ok = verify_audio("test_data/reference.wav", 2, 10, "Reference")

    # Verify training audio (few-shot)
    train_ok = verify_audio("test_data/training.wav", 3, 120, "Training")

    if not (ref_ok and train_ok):
        print("\n❌ Audio verification failed!")
        exit(1)
    
    print("\n✓ All audio files verified successfully!")

if __name__ == "__main__":
    main()
