import ollama
import logging
import traceback

class LLMService:
    """A service for interacting with a local LLM via Ollama."""

    def __init__(self, model='gemma3:12b-it-qat'):
        """
        Initializes the LLM service.
        :param model: The name of the Ollama model to use.
        """
        logging.info(f"Initializing LLMService with model '{model}'...")
        self._model = model
        self._client = ollama.Client()

        # Verify the model is available locally
        try:
            self._client.show(model)
            logging.info(f"Model '{model}' is available locally.")
        except ollama.ResponseError as e:
            logging.error(f"Model '{model}' not found locally. Please run 'ollama pull {model}'.")
            raise e

    def generate_response(self, conversation_history: list) -> str:
        """
        Sends a conversation history to the Ollama model and gets a response.
        Ensures the final output is always a string.
        """
        try:
            logging.info("Sending prompt to LLM...")
            response = ollama.chat(model=self._model, messages=conversation_history)
            response_content = response['message']['content']

            # --- FIX: Force output to be a string ---
            if isinstance(response_content, list):
                # If the model returns a list of strings, join them.
                logging.warning("LLM returned a list, joining to string.")
                return " ".join(map(str, response_content))

            # Ensure it's a string just in case it's some other type.
            return str(response_content)

        except Exception as e:
            logging.error(f"Error communicating with LLM: {e}")
            logging.error(traceback.format_exc())
            return "I'm sorry, I encountered an error and couldn't process your request."

if __name__ == '__main__':
    # Example usage of the service
    print("--- Testing LLMService ---")
    llm = LLMService()

    # This is the corrected conversation history definition.
    # It's a list of message dictionaries for the test.
    conversation_history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello! Tell me a short, interesting fact about the planet Mars.'}
    ]

    response_text = llm.generate_response(conversation_history)
    print(f"Assistant: {response_text}")
    print("--- Test Complete ---")
