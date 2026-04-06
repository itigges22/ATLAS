"""Solve command — generate, verify, and test a coding solution."""

import re
import time

from atlas.cli import display, client


def format_prompt(problem: str) -> str:
    """Format problem as ChatML prompt."""
    return (
        f"<|im_start|>system\n"
        f"You are a competitive programming expert. Write clean, correct Python code "
        f"that reads from stdin and prints to stdout. Think step by step.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{problem}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_code(text: str) -> str:
    """Extract code from model response."""
    clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "</think>" in clean and "<think>" not in clean:
        clean = clean[clean.index("</think>") + len("</think>"):].strip()
    if "<think>" in clean:
        # Unclosed think — take content after it if it has code
        after = clean[clean.index("<think>") + len("<think>"):].strip()
        before = clean[:clean.index("<think>")].strip()
        if ("def " in after or "class " in after) and "def " not in before:
            clean = after
        elif "def " in before:
            clean = before

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", clean, re.DOTALL)
    if blocks:
        return blocks[0].strip()

    for line in clean.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            idx = clean.index(line)
            return clean[idx:].strip()

    return clean.strip()


def solve(problem: str, stream: bool = True, verify: bool = True,
          sandbox_test: str = ""):
    """Solve a coding problem with the full ATLAS pipeline."""
    start = time.time()
    prompt = format_prompt(problem)

    display.assistant_label()

    if stream:
        full_text = ""
        in_think = False
        think_started = False
        think_done = False
        code_started = False
        tokens = 0

        try:
            for token_text, is_done in client.generate_stream(
                prompt, max_tokens=8192, stop=["<|im_end|>"]
            ):
                full_text += token_text
                tokens += 1

                # Detect thinking start
                if "<think>" in full_text and not think_started:
                    think_started = True
                    in_think = True
                    display.stream_thinking_start()
                    # Don't display the <think> tag itself
                    continue

                # Detect thinking end
                if "</think>" in token_text and in_think:
                    think_done = True
                    in_think = False
                    display.stream_thinking_end()
                    continue

                # Stream tokens
                if in_think:
                    # Skip the literal <think> and </think> tags
                    clean = token_text.replace("<think>", "").replace("</think>", "")
                    if clean:
                        display.stream_thinking_token(clean)
                elif think_done and not code_started:
                    # After thinking, look for code
                    stripped = token_text.strip()
                    if stripped and stripped not in ("\n", "\n\n"):
                        code_started = True
                        display.stream_code_start()
                        display.stream_code_token(token_text.lstrip("\n"))
                elif code_started:
                    display.stream_code_token(token_text)
                else:
                    # No thinking block — stream directly as code
                    if not code_started and token_text.strip():
                        code_started = True
                        display.stream_code_start()
                    if code_started:
                        display.stream_code_token(token_text)

                if is_done:
                    break

            if code_started:
                display.stream_code_end()

        except Exception as e:
            display.warn(f"Stream error: {e}")
            # Fallback to batch
            data = client.generate(prompt, max_tokens=8192, stop=["<|im_end|>"])
            full_text = data["choices"][0].get("text", "")
            tokens = data.get("usage", {}).get("completion_tokens", 0)
    else:
        # Batch mode
        data = client.generate(prompt, max_tokens=8192, stop=["<|im_end|>"])
        full_text = data["choices"][0].get("text", "")
        tokens = data.get("usage", {}).get("completion_tokens", 0)

    # Extract code
    code = extract_code(full_text)
    if not code:
        elapsed = time.time() - start
        display.solution_failed(tokens, elapsed, "no code extracted")
        return None

    # Verification
    if verify:
        display.phase_label("Verifying")
        energy, normalized = client.score_code(code)
        display.energy_score(energy, normalized)

    # Sandbox
    if sandbox_test:
        display.phase_label("Testing")
        passed, stdout, stderr = client.run_sandbox(code, sandbox_test)
        display.sandbox_result(passed, stderr[:80] if stderr else "")

    elapsed = time.time() - start
    display.solution_accepted(tokens, elapsed)
    return code


def solve_file(filepath: str, **kwargs):
    """Solve from a file."""
    with open(filepath) as f:
        problem = f.read()
    return solve(problem, **kwargs)
