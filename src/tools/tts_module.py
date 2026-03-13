import pyttsx3
import os
import uuid
import config


def generate_audio_report(text: str) -> str:
    """
    Converts the final text report into an audio file using local CPU TTS.
    Returns the path to the saved audio file.
    """
    print("--- GENERATING OFFLINE AUDIO REPORT ---")
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 0.9)

    clean_text = text.replace("**", "").replace("*", "").replace("#", "")

    filename = f"explanation_{uuid.uuid4().hex[:6]}.wav"
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    engine.save_to_file(clean_text, filepath)
    engine.runAndWait()

    print(f"✅ Audio report saved to: {filepath}")
    return filepath
