import logfire

# Use this to set the model used in the experiments
MODEL = 'anthropic:claude-haiku-4-5-20251001'

# ---------------------------------------------------------------------------
# Logfire
# ---------------------------------------------------------------------------

def configure_logfire() -> None:
    """Configure logfire and instrument pydantic-ai (no-op when no token is set)."""
    logfire.configure(send_to_logfire='if-token-present')
    logfire.instrument_pydantic_ai()
