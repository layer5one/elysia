import logging
import os
import datetime
import traceback

# Import our services
from stt_service import SpeechToTextService
from tts_service import TextToSpeechService
from llm_service import LLMService

# --- CHOOSE YOUR MEMORY SYSTEM ---
# Option A: Stable ChromaDB
from memory_service_chroma import ChromaMemoryService as MemoryService
# Option B: Adventurous MemOS
# from memory_service_memos import MemOSMemoryService as MemoryService
# ---------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalAI:
    def __init__(self):
        self.llm = LLMService()
        self.stt = SpeechToTextService()
        self.tts = TextToSpeechService()
        self.memory = MemoryService()
        self.persona_prompt = (
            "You are a helpful, first-person conversational AI assistant. "
            "Your responses should be detailed and informative. "
            "You have a memory of past conversations and can refer to them. "
            "You also have a special ability: when you provide a detailed response, "
            "you also create a very short, one or two-sentence summary of it that you will speak aloud. "
            "You save the full detailed response to a text file. Be aware of the files you have saved."
        )
        self.response_log_dir = "response_logs"
        os.makedirs(self.response_log_dir, exist_ok=True)

    def _muzzle_and_save(self, full_response: str) -> tuple[str, str]:
        """
        Summarizes the full response for TTS and saves the full response to a file.
        """
        logging.info("Muzzling response and saving full text...")

        # This is the corrected and completed prompt definition.
        # It instructs the LLM how to summarize the text for spoken output.
        summarization_prompt = [
            {
                'role': 'system',
                'content': 'You are an expert in summarization. Your task is to take the following text and create a concise, one or two-sentence summary suitable for being spoken aloud. Do not add any conversational fluff. Just provide the summary.'
            },
            {
                'role': 'user',
                'content': full_response
            }
        ]

        spoken_summary = self.llm.generate_response(summarization_prompt)

        # Save the full response to a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.response_log_dir, f"response_{timestamp}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_response)

        logging.info(f"Spoken summary: '{spoken_summary}'")
        logging.info(f"Full response saved to: '{file_path}'")

        return spoken_summary, file_path

    def run(self):
        self.tts.speak("System online. I am ready.")
        while True:
            try:
                # 1. Listen for user input
                user_input = self.stt.listen()
                if not user_input:
                    continue

                # 2. Retrieve relevant memories
                relevant_memories = self.memory.retrieve_relevant_memories(user_input)
                memory_context = "\n".join(relevant_memories)

                # 3. Construct the prompt for the LLM
                prompt_messages = [
                    {'role': 'system', 'content': self.persona_prompt},
                    {'role': 'system', 'content': f"Here is some relevant context from our past conversations:\n{memory_context}"},
                    {'role': 'user', 'content': user_input}
                ]

                # 4. Generate the full, detailed response
                full_response = self.llm.generate_response(prompt_messages)

                # 5. Muzzle the response: get spoken summary and file path
                spoken_summary, file_path = self._muzzle_and_save(full_response)

                # 6. Speak the summary
                self.tts.speak(spoken_summary)

                # 7. Update memory with the full exchange and file context
                self.memory.add_memory(user_input, full_response)
                self.memory.add_system_memory(
                    f"(Self-reflection: I just answered the user's query about '{user_input}'. "
                    f"I spoke a summary and saved the full details to the file '{file_path}'.)"
                )

            except KeyboardInterrupt:
                logging.info("Shutdown signal received.")
                self.tts.speak("Shutting down. Goodbye.")
                break

            except Exception as e:
                # This will now print the full error traceback to the console
                logging.error(f"An error occurred in the main loop: {e}")
                logging.error(f"Error Type: {type(e)}")
                logging.error("--- FULL TRACEBACK ---")
                logging.error(traceback.format_exc())
                logging.error("----------------------")
                self.tts.speak("I've encountered an internal error. Please check the logs.")


if __name__ == '__main__':
    ai = ConversationalAI()
    ai.run()
