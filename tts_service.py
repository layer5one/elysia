from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import traceback
import logging
import time  # NEW: timestamps for stream events

# NEW: WS broadcaster for UI
try:
    from tts_ws import WS
except Exception:
    WS = None
    logging.warning("tts_ws.WS not available; UI streaming disabled.")

class TextToSpeechService:
    """A service for generating high-quality speech from text."""

    def __init__(self):
        logging.info("Initializing TextToSpeechService...")
        # Suppress excessive warnings from underlying libraries
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)

        self.voice = "af_heart"
        self.sample_rate = 24000  # Hardcoded to match the library's requirement

        try:
            # Provide the library's unique code for American English.
            self.engine = KPipeline(lang_code="a")

            logging.info(f"TextToSpeechService initialized for American English ('a').")
            logging.info(f"Sample rate set to {self.sample_rate} Hz.")

        except Exception as e:
            logging.error(f"Failed to initialize Kokoro TTS engine: {e}")
            logging.error(traceback.format_exc())
            raise

    def speak(self, text: str):
        """
        Generates audio from text and plays it using sounddevice.
        Also streams PCM chunks over WS for the UI (if WS is available).
        """
        if not text:
            return

        try:
            logging.info(f"TTS generating audio for: '{text}'")
            audio_chunks = []  # keep your original name

            msg_id = f"msg_{int(time.time()*1000)}"
            if WS: WS.tts_begin(self.sample_rate, msg_id)

            # KPipeline yields Result objects which contain the audio (torch.FloatTensor)
            for result in self.engine(text=text, voice=self.voice):
                if result.audio is not None:
                    # Debug prints you had before (optional)
                    # print(">>> TTS DEBUG — result.audio =", type(result.audio))
                    # print(">>> requires_grad:", getattr(result.audio, "requires_grad", "N/A"))
                    # print(">>> is_cuda:", getattr(result.audio, "is_cuda", "N/A"))
                    # print(">>> device:", getattr(result.audio, "device", "N/A"))

                    # Convert to numpy float32 for sounddevice and WS
                    audio_chunk = result.audio.detach().cpu().numpy().astype(np.float32)
                    audio_chunks.append(audio_chunk)

                    # Stream to UI if available
                    if WS:
                        WS.tts_chunk(msg_id, time.time(), audio_chunk.tobytes())

            if not audio_chunks:
                logging.warning("TTS generated no audio chunks.")
                if WS: WS.tts_end(msg_id)
                return

            full_audio = np.concatenate(audio_chunks, axis=0)
            sd.play(full_audio, self.sample_rate)
            sd.wait()

            if WS: WS.tts_end(msg_id)

        except Exception as e:
            logging.error(f"Error in TTS service: {e}")
            logging.error(traceback.format_exc())
            if WS: WS.state("error")

if __name__ == '__main__':
    # Example usage
    tts = TextToSpeechService()

    # Example with emphasis control
    expressive_text = "I [really](+1) think this is a [fantastic](+2) idea. Let's do it."
    tts.speak(expressive_text)

    # Example with pronunciation control
    pronunciation_text = "The model is called [Kokoro](/kˈOkəɹO/)."
    tts.speak(pronunciation_text)
