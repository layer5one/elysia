import logging
import os
import datetime
import traceback

# Import our services
from stt_service import SpeechToTextService
from tts_service import TextToSpeechService
from llm_service import LLMService
from tool_service import ToolService   # NEW: import the ToolService

# --- CHOOSE YOUR MEMORY SYSTEM ---
from memory_service_chroma import ChromaMemoryService as MemoryService
# Option B: Adventurous MemOS (currently not in use)
# from memory_service_memos import MemOSMemoryService as MemoryService
# ---------------------------------

# Configure logging (log to file as well for crash tracking)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("elysia.log", mode='a', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

class ConversationalAI:
    def __init__(self):
        self.llm = LLMService()
        self.stt = SpeechToTextService()
        self.tts = TextToSpeechService()
        self.memory = MemoryService()
        self.tool = ToolService()   # NEW: initialize the ToolService

        # Extended persona prompt with tool instructions
        tool_guide = (
            "TOOLS AVAILABLE: \n"
            "You can use the following tools by responding with a line starting with 'TOOL:':\n"
            "- read_file(\"<path>\"): Read a file and get its content.\n"
            "- write_file(\"<path>\", \"<content>\"): Create or overwrite a file with given content.\n"
            "- execute_python(\"\"\"\n<code>\n\"\"\"): Execute Python code and get the output.\n"
            "- execute_shell(\"<command>\"): Run a shell command and get its output.\n"
            "- use_gemini(\"<query>\"): Ask an advanced AI (Gemini) for help on the query.\n"
            "If you use a tool, respond exactly with the format: TOOL: <ToolName>(args). I will execute it and provide the result.\n"
            "Only provide a final answer when you have gathered all necessary information."
        )
        self.persona_prompt = (
            "You are Elysia, a powerful first-person conversational AI assistant. "
            "You have access to tools that you can use to better answer the user's requests. "
            "Your responses should be detailed and informative. You have a memory of past conversations. "
            "Always consider using a tool if it will improve your answer or accuracy.\n"
            + tool_guide
        )
        self.response_log_dir = "response_logs"
        os.makedirs(self.response_log_dir, exist_ok=True)

        # Check if there is crash info from a previous run
        if os.path.exists("crash_info.txt"):
            try:
                crash_details = open("crash_info.txt", "r").read()
                # Add a system memory note about the crash
                self.memory.add_system_memory(f"(Last crash report: {crash_details})")
                logging.info("Loaded crash info into system memory.")
                # Optionally, remove or archive the crash_info.txt after reading
                os.remove("crash_info.txt")
            except Exception as e:
                logging.error(f"Could not load crash info: {e}")

    def _muzzle_and_save(self, full_response: str) -> tuple[str, str]:
        """
        Summarizes the full response for TTS and saves the full response to a file.
        """
        logging.info("Muzzling response and saving full text...")
        # Summarization prompt for TTS
        summarization_prompt = [
            {'role': 'system', 'content': 'You are an expert summarizer. Provide a concise one or two sentence summary of the assistant response for speaking aloud.'},
            {'role': 'user', 'content': full_response}
        ]
        try:
            spoken_summary = self.llm.generate_response(summarization_prompt)
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            # Fallback: truncate full_response for speech if summarization fails
            spoken_summary = (full_response[:150] + '...') if len(full_response) > 150 else full_response

        # Save the full response to a timestamped file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.response_log_dir, f"response_{timestamp}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_response)
        logging.info(f"Full response saved to: '{file_path}'")

        logging.info(f"Spoken summary: '{spoken_summary}'")
        return spoken_summary, file_path

    def run(self):
        # Announce system readiness
        self.tts.speak("System online. I am ready.")
        logging.info("Elysia is now running. Waiting for user input...")

        while True:
            try:
                # 1. Listen for user input (via STT)
                user_input = self.stt.listen()
                if not user_input:
                    continue  # if nothing heard, loop again

                logging.info(f"User said: {user_input}")

                # 2. Retrieve relevant memories for context
                relevant_memories = self.memory.retrieve_relevant_memories(user_input)
                memory_context = "\n".join(relevant_memories)

                # 3. Construct initial prompt messages
                messages = [
                    {'role': 'system', 'content': self.persona_prompt},
                    {'role': 'system', 'content': f"Relevant past context:\n{memory_context}"},
                    {'role': 'user', 'content': user_input}
                ]

                # 4. Interaction loop with LLM (tool usage or final answer)
                full_response = ""   # will accumulate the assistant's answer
                step_count = 0
                while True:
                    step_count += 1
                    logging.info(f"LLM thinking... (step {step_count})")
                    # Generate a response from the LLM given the current conversation state
                    assistant_reply = self.llm.generate_response(messages)
                    if assistant_reply is None:
                        assistant_reply = ""

                    assistant_reply = assistant_reply.strip()
                    logging.info(f"LLM reply: {assistant_reply}")

                    # Check if the LLM wants to use a tool (indicated by the special prefix)
                    if assistant_reply.lower().startswith("tool:"):
                        # Parse tool name and arguments from the reply
                        # Expected format example: "TOOL: read_file(\"path/to/file.txt\")"
                        tool_call = assistant_reply[len("tool:"):].strip()
                        # Extract the tool name and params
                        # Simple parsing: tool_name is before the first '('
                        tool_name = tool_call.split('(', 1)[0].strip()
                        args_str = tool_call[len(tool_name):].strip().lstrip('(').rstrip(')')
                        # Remove any surrounding quotes from args and split by comma if needed
                        # (This is a simplistic parser; assumes no nested commas in args)
                        args = []
                        if args_str:
                            try:
                                # Try to eval the arguments string as Python literal tuple
                                parsed = eval(f"({args_str})")
                                # If a single arg, eval returns (arg,), so handle that
                                if isinstance(parsed, tuple):
                                    args = list(parsed)
                                else:
                                    args = [parsed]
                            except Exception as e:
                                # Fallback: just use raw string (strip quotes)
                                cleaned = args_str.strip().strip('"').strip("'")
                                if cleaned:
                                    args = [cleaned]
                        # Execute the corresponding tool function
                        result = ""
                        try:
                            if tool_name == "read_file" and len(args) == 1:
                                result = self.tool.read_file(args[0])
                            elif tool_name == "write_file" and len(args) == 2:
                                filepath, content = args[0], args[1]
                                result = self.tool.write_file(filepath, content)
                            elif tool_name == "execute_python" and len(args) == 1:
                                code = args[0]
                                result = self.tool.execute_python(code)
                            elif tool_name == "execute_shell" and len(args) == 1:
                                command = args[0]
                                result = self.tool.execute_shell(command)
                            elif tool_name == "use_gemini" and len(args) == 1:
                                query = args[0]
                                result = self.tool.use_gemini(query)
                            else:
                                result = f"[Error: Unknown tool or wrong arguments: {tool_name}]"
                        except Exception as e:
                            result = f"[Error: Exception during tool execution: {e}]"
                        # Log the tool execution result
                        logging.info(f"Tool {tool_name} executed. Result: {result[:200]}...")  # log first 200 chars
                        # Append the tool result as a system message for the LLM
                        messages.append({'role': 'system', 'content': f"Tool output:\n{result}"})
                        # Continue the loop to let LLM process this new info
                        if step_count >= 5:  # safety break to avoid infinite loops
                            logging.warning("Too many tool steps, breaking out to prevent loop.")
                            break
                        else:
                            continue
                    else:
                        # No tool usage indicated, so this should be the final answer
                        full_response = assistant_reply
                        break

                # 5. Muzzle the response: get spoken summary and file path
                spoken_summary, file_path = self._muzzle_and_save(full_response)

                # 6. Speak the summary out loud
                self.tts.speak(spoken_summary)

                # 7. Update memory with the full exchange and any system notes
                self.memory.add_memory(user_input, full_response)
                self.memory.add_system_memory(
                    f"(Self-reflection: Provided a detailed answer. Full response saved to '{file_path}'.)"
                )
                # If any tool was used, optionally we could add a memory note about that as well
                # e.g., self.memory.add_system_memory("(Used tools to assist in the above answer.)")

            except KeyboardInterrupt:
                logging.info("Shutdown signal received. Exiting.")
                self.tts.speak("Shutting down. Goodbye.")
                break

            except Exception as e:
                # Log the exception with traceback
                logging.error(f"An error occurred in the main loop: {e}")
                logging.error("--- FULL TRACEBACK ---\n" + traceback.format_exc())
                # Inform the user via speech (briefly)
                self.tts.speak("I've encountered an internal error and will attempt to recover.")
                # Write crash info to file for watchdog and next session
                with open("crash_info.txt", "w") as cf:
                    cf.write(f"{e}\n{traceback.format_exc()}")
                # Break out of the loop so that watchdog (if running) can restart the process
                break

if __name__ == '__main__':
    ai = ConversationalAI()
    ai.run()
