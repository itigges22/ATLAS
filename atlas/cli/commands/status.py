"""Status command — check health of all ATLAS services."""

from atlas.cli import display, client


def status():
    """Check and display health of all services."""
    display.separator()

    # vLLM
    ok, detail = client.check_llama()
    if ok:
        display.success(f"vLLM: {detail}")
    else:
        display.error(f"vLLM: {detail}")

    # geometric-lens
    ok, detail = client.check_rag_api()
    if ok:
        display.success(f"Lens: {detail}")
    else:
        display.error(f"Lens: {detail}")

    # Sandbox
    ok, detail = client.check_sandbox()
    if ok:
        display.success(f"Sandbox: {detail}")
    else:
        display.error(f"Sandbox: {detail}")

    display.separator()
