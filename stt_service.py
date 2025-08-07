from RealtimeSTT import AudioToTextRecorder
import logging
import traceback

class SpeechToTextService:
    """A service for real-time speech-to-text transcription."""

    def __init__(self):
        logging.info("Initializing SpeechToTextService...")

        # Define the model and language settings first.
        self.model = "tiny.en"
        self.language = "en"

        try:
            # Now, use these attributes to initialize the recorder.
            self.recorder = AudioToTextRecorder(
                model=self.model,
                language=self.language,
                early_transcription_on_silence=2000,
            )
            logging.info(f"SpeechToTextService initialized with model '{self.model}'.")

        except Exception as e:
            logging.error(f"Failed to initialize AudioToTextRecorder: {e}")
            logging.error(traceback.format_exc())
            raise

    def _on_wakeword(self):
        print("Wake word detected! Listening for your command...")

    def _on_record_start(self):
        print("Recording started...")

    def _on_record_stop(self):
        print("Recording stopped.")

    def listen(self) -> str:
        """
        Listens for a single utterance from the user and returns the transcription.
        This is a blocking call.
        """
        print("Listening for wake word...")
        # The text() method with no callback blocks until a transcription is complete.
        transcription = self.recorder.text()
        print(f"Transcription: '{transcription}'")
        return transcription

if __name__ == '__main__':
    # Example usage of the service
    stt = SpeechToTextService()
    while True:
        user_input = stt.listen()
        if "exit" in user_input.lower():
            print("Exit command received. Shutting down.")
            break
