# llm_service.py
import llm
import logging

DEFAULT_MODEL = "elysia"  # your Ollama model name; change if needed

class LLMService:
    """LLM wrapper using Simon Willison's `llm` Python API, with tool support."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        logging.info(f"Initializing LLMService via llm: model='{self.model_id}'")
        try:
            self.model = llm.get_model(self.model_id)
        except Exception as e:
            logging.error(f"llm.get_model('{self.model_id}') failed: {e}")
            raise

    def prompt(self, prompt: str, system: str | None = None, tools: list | None = None) -> str:
        """
        One-shot prompt. If tools provided, model may call them; use chain() for auto-return.
        """
        resp = self.model.prompt(prompt, system=system or "", tools=tools or [])
        # If tools were requested, caller can handle resp.tool_calls() etc.
        return resp.text()

    def chain(self, prompt: str, system: str | None = None, tools: list | None = None) -> str:
        """
        Run an agent-style chain (model can call tools; results fed back automatically).
        Returns final stitched text.
        """
        chain = self.model.chain(prompt, system=system or "", tools=tools or [])
        # Gather the full final text
        chunks = []
        for chunk in chain:
            chunks.append(chunk)
        return "".join(chunks)
