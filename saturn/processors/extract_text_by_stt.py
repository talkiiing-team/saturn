from saturn.libs.broker import broker

speech_to_text = None
if broker.is_worker_process:
    import whisper

    speech_to_text = whisper.load_model("tiny", "cuda")


def extract_text_by_stt(audio_path: str) -> str:
    try:
        speech_text = speech_to_text.transcribe(audio_path, language="ru")["text"].strip()
    except Exception:
        speech_text = ""

    return speech_text or ""
