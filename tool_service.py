# tool_service.py
import os, io, shlex, subprocess, contextlib, traceback, time, json
import llm  # needed for Toolbox base

PROJECT_ROOT = os.path.abspath(os.getcwd())
MAX_READ_BYTES = 5 * 1024 * 1024  # 5MB cap per read
MAX_ECHO_CHARS = 10000            # cap what we return to model
GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"

def _safe_path(p: str) -> str:
    p = os.path.abspath(os.path.expanduser(p))
    # keep within project root (relax if you want)
    if not p.startswith(PROJECT_ROOT):
        raise ValueError("Path outside project root not allowed")
    return p

class ElysiaTools(llm.Toolbox):
    """
    Multi-tool box available to the model via LLM tool calling.
    Methods are tools. Keep docstrings tight—they become tool docs.
    """

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from disk and return its content (truncated if huge)."""
        fp = _safe_path(path)
        if not os.path.exists(fp):
            return "[Error: file not found]"
        if os.path.getsize(fp) > MAX_READ_BYTES:
            return "[Error: file too large]"
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            if len(data) > MAX_ECHO_CHARS:
                return data[:MAX_ECHO_CHARS] + "\n[Content truncated]"
            return data
        except Exception as e:
            return f"[Error reading file: {e}]"

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file with provided content. Backs up existing as .bak."""
        fp = _safe_path(path)
        try:
            if os.path.exists(fp):
                bk = fp + ".bak"
                try:
                    os.replace(fp, bk)
                except Exception:
                    import shutil; shutil.copy(fp, bk)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[Wrote {len(content)} chars to {fp}]"
        except Exception as e:
            return f"[Error writing file: {e}]"

    def append_file(self, path: str, content: str) -> str:
        """Append text to a UTF-8 file, creating it if missing."""
        fp = _safe_path(path)
        try:
            with open(fp, "a", encoding="utf-8") as f:
                f.write(content)
            return f"[Appended {len(content)} chars to {fp}]"
        except Exception as e:
            return f"[Error appending file: {e}]"

    def execute_python(self, code: str) -> str:
        """Execute Python code in-process; return stdout or error traceback. Use `result = ...` to return values."""
        try:
            buf = io.StringIO()
            ns = {}
            with contextlib.redirect_stdout(buf):
                try:
                    exec(code, ns)
                except Exception:
                    return "[Error during execution]\n" + traceback.format_exc()
            out = buf.getvalue()
            if (not out.strip()) and ("result" in ns):
                out = str(ns["result"])
            if not out.strip():
                out = "[Executed successfully with no output]"
            return out[:MAX_ECHO_CHARS] + ("..." if len(out) > MAX_ECHO_CHARS else "")
        except Exception:
            return "[Error: execution failed]\n" + traceback.format_exc()

    def execute_shell(self, command: str) -> str:
        """Run a shell command with a short timeout; return stdout or stderr."""
        try:
            p = subprocess.run(command, shell=True, text=True,
                               capture_output=True, timeout=20)
            if p.returncode != 0:
                err = (p.stderr or "").strip()
                if not err:
                    err = f"non-zero exit ({p.returncode})"
                return f"[Shell error] {err}"
            out = (p.stdout or "").strip()
            return out[:MAX_ECHO_CHARS] + ("..." if len(out) > MAX_ECHO_CHARS else "") or "[ok]"
        except subprocess.TimeoutExpired:
            return "[Shell error] timed out"
        except Exception as e:
            return f"[Shell error] {e}"

    def gemini_cli(self, prompt: str, model: str = GEMINI_DEFAULT_MODEL) -> str:
        """Delegate to GeminiCLI for complex drafting or function creation. Requires GEMINI_API_KEY and `gemini` on PATH."""
        if not os.getenv("GEMINI_API_KEY"):
            return "[GeminiCLI error] GEMINI_API_KEY not set"
        cmd = f'gemini -p {shlex.quote(prompt)} -m {shlex.quote(model)}'
        try:
            p = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=120)
            if p.returncode != 0:
                return f"[GeminiCLI error] {(p.stderr or 'non-zero exit').strip()}"
            out = (p.stdout or "").strip()
            return out[:MAX_ECHO_CHARS] + ("..." if len(out) > MAX_ECHO_CHARS else "") or "[GeminiCLI: no output]"
        except subprocess.TimeoutExpired:
            return "[GeminiCLI error] timed out"
        except Exception as e:
            return f"[GeminiCLI error] {e}"
