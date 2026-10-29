class ToolError(Exception):
    """Typed, explicit tool failure. Surfaced to the LLM as an error payload, never as data."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def to_payload(self) -> dict:
        return {"error": self.code, "message": self.message}
