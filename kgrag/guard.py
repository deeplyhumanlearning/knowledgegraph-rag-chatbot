from nemoguardrails import LLMRails, RailsConfig

from kgrag.config import GUARDRAILS_DIR

REFUSAL = "Sorry, I can only help with questions about Tamil Nadu government schemes."

rails = LLMRails(RailsConfig.from_path(str(GUARDRAILS_DIR)))


def _passes(messages: list[dict], rail: str) -> bool:
    """Run a single NeMo rail; it echoes the last message back unchanged when allowed."""
    result = rails.generate(messages=messages, options={"rails": [rail]})
    return result.response[-1]["content"] == messages[-1]["content"]


def check_input(question: str) -> bool:
    return _passes([{"role": "user", "content": question}], "input")


def check_output(question: str, answer: str) -> bool:
    return _passes([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "output")
