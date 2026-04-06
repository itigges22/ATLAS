"""Status command — check health of all ATLAS services."""

from atlas.cli import display, client


def status():
    """Check and display health of all services."""
    display.separator()

    # Fox
    ok, detail = client.check_fox()
    if ok:
        display.success(f"Fox: {detail}")
    else:
        display.error(f"Fox: {detail}")

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
