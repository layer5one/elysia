# main_app.py
import logging, os, datetime, traceback

from stt_service import SpeechToTextService
from tts_service import TextToSpeechService
from llm_service import LLMService
from memory_service_chroma import ChromaMemoryService as MemoryService
from tool_service import ElysiaTools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("elysia.log", mode="a", encoding="utf-8"),
              logging.StreamHandler()]
)

class ConversationalAI:
    def __init__(self):
        self.llm = LLMService()  # uses llm.get_model("elysia")
        self.stt = SpeechToTextService()
        self.tts = TextToSpeechService()
        self.memory = MemoryService()
        self.tools = ElysiaTools()  # llm toolbox (read/write/exec/gemini_cli)
        self.response_log_dir = "response_logs"
        os.makedirs(self.response_log_dir, exist_ok=True)

        # Persona + capability note (keep short; details in memory context)
        self.persona_prompt = (
            "You are Elysia: blunt, high-context, tool-using local AI. "
            "Use tools when they improve accuracy or enable real action. "
            "Prefer concrete steps over vague generalities. Avoid corporate tone."
        )

        # Ingest prior crash info (from watchdog or last run)
        if os.path.exists("crash_info.txt"):
            try:
                crash = open("crash_info.txt", "r", encoding="utf-8").read()
                self.memory.add_system_memory(f"(Last crash report captured on restart)\n{crash}")
                logging.info("Loaded crash_info.txt into memory and removed it.")
                os.remove("crash_info.txt")
            except Exception as e:
                logging.error(f"Failed to load crash_info.txt: {e}")

    def _muzzle_and_save(self, full_response: str) -> tuple[str, str]:
        """
        Create a one/two-sentence spoken summary and save the full response to file.
        If summarization fails, speak a truncation.
        """
        # Summarize using the same model (no tools)
        try:
            summary = self.llm.prompt(
                "Summarize succinctly in 1-2 sentences for speech:\n\n" + full_response,
                system="Be direct, no fluff."
            )
        except Exception as e:
            logging.error(f"Summary failed: {e}")
            summary = (full_response[:150] + "...") if len(full_response) > 150 else full_response

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(self.response_log_dir, f"response_{ts}.txt")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(full_response)
        logging.info(f"Saved full response to {fp}")
        return summary, fp

    def run(self):
        self.tts.speak("System online. Ready.")
        logging.info("Elysia running.")

        while True:
            try:
                # 1) STT
                user_input = self.stt.listen()
                if not user_input:
                    continue
                logging.info(f"USER: {user_input}")

                # 2) Memory retrieval (hybrid could be extended later)
                memories = self.memory.retrieve_relevant_memories(user_input)
                memory_context = "\n".join(memories)

                # 3) Build system + prompt (short system, stuff context in system too)
                system = self.persona_prompt + "\n\nRelevant context:\n" + (memory_context or "[none]")
                prompt = user_input

                # 4) Chain with tools (model decides to call them)
                full_response = self.llm.chain(prompt, system=system, tools=[self.tools])

                # 5) Speak short; save long
                spoken_summary, path = self._muzzle_and_save(full_response)
                self.tts.speak(spoken_summary)

                # 6) Persist memory
                self.memory.add_memory(user_input, full_response)
                self.memory.add_system_memory(f"(Saved full response to {path})")

            except KeyboardInterrupt:
                logging.info("Shutdown requested.")
                self.tts.speak("Shutting down. Goodbye.")
                break
            except Exception as e:
                logging.error("Fatal error in main loop: %s", e)
                logging.error("--- TRACEBACK ---\n" + traceback.format_exc())
                self.tts.speak("Encountered an internal error. Attempting recovery.")
                # write crash info for watchdog + next boot
                with open("crash_info.txt", "w", encoding="utf-8") as cf:
                    cf.write(str(e) + "\n" + traceback.format_exc())
                break  # let watchdog restart

if __name__ == "__main__":
    ConversationalAI().run()
