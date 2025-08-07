import os, subprocess, io, contextlib, traceback
from google import genai

# Initialize the Gemini API client once (assumes GEMINI_API_KEY is set in environment)
try:
    gemini_client = genai.Client() 
except Exception as e:
    gemini_client = None
    print("Warning: Gemini client initialization failed. Ensure GEMINI_API_KEY is set. Error:", e)

class ToolService:
    """Provides various tool functions that the AI can use for extended capabilities."""
    
    def __init__(self):
        # You can initialize any state or allowed paths here
        self.working_dir = os.getcwd()
    
    def read_file(self, filepath: str) -> str:
        """Read and return the content of the given file."""
        try:
            # Security check: optionally, prevent reading very large files or binary files
            if os.path.getsize(filepath) > 5 * 1024 * 1024:  # 5 MB limit for example
                return "[Error: File too large to read]"
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Truncate content if extremely long to avoid overloading the model context
            if len(content) > 10000:  # e.g., 10k characters limit
                return content[:10000] + "\n[Content truncated]"
            return content
        except FileNotFoundError:
            return "[Error: File not found]"
        except Exception as e:
            return f"[Error: Could not read file: {e}]"
    
    def write_file(self, filepath: str, new_content: str) -> str:
        """Write the given content to the file. Creates or overwrites the file. Returns a status message."""
        try:
            # Backup existing file if it exists
            if os.path.exists(filepath):
                backup_path = filepath + ".bak"
                try:
                    os.replace(filepath, backup_path)
                except Exception as e:
                    # If replace fails (e.g., cross-filesystem), use copy
                    import shutil
                    shutil.copy(filepath, backup_path)
                msg = f"(Backed up original to {backup_path}). "
            else:
                msg = ""
            # Write new content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return msg + f"Successfully wrote to {filepath}."
        except Exception as e:
            return f"[Error: Could not write to file: {e}]"
    
    def execute_python(self, code: str) -> str:
        """Execute the given Python code and return the stdout and/or error output."""
        try:
            # Redirect stdout to capture prints
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                exec_namespace = {}
                try:
                    exec(code, exec_namespace)
                except Exception as e:
                    # Capture traceback
                    traceback_str = traceback.format_exc()
                    return f"[Error during execution]\n{traceback_str}"
            output = stdout_buffer.getvalue()
            if output.strip() == "" and 'result' in exec_namespace:
                # If code defines a variable 'result', return it (as string)
                output = str(exec_namespace['result'])
            return output if output else "[Executed successfully with no output]"
        except Exception as e:
            err_trace = traceback.format_exc()
            return f"[Error: Execution failed]\n{err_trace}"
    
    def execute_shell(self, command: str) -> str:
        """Execute a shell command and return its output (stdout or error). Use with caution!"""
        try:
            completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
            out = completed.stdout.strip()
            err = completed.stderr.strip()
            if completed.returncode != 0 or err:
                return f"[Shell error] {err if err else 'Command returned non-zero status.'}"
            return out if out else "[Command executed successfully with no output]"
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out]"
        except Exception as e:
            return f"[Error: Failed to execute shell command: {e}]"
    
    def use_gemini(self, prompt: str) -> str:
        """Send a prompt to the Gemini model (via API) and return its response."""
        if gemini_client is None:
            return "[Error: Gemini client not initialized or API key missing]"
        try:
            response = gemini_client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
            result = response.text
            # Truncate extremely long Gemini responses for safety
            if len(result) > 10000:
                result = result[:10000] + "\n[Response truncated]"
            return result
        except Exception as e:
            # If any error occurs (e.g., rate limit, API fail), return a message
            return f"[Error: Gemini API call failed: {e}]"
