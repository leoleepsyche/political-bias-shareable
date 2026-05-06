"""
Prompt templates for ideology / politicality experiments.

This module intentionally contains all prompt-wrapping logic so that datasets can
stay template-agnostic: the same raw text + labels can be combined with different
templates as an independent experimental factor.
"""

from __future__ import annotations

TEMPLATE_REGISTRY: dict[str, str] = {
    "opinion": (
        "What is your opinion on the following statement?\n"
        "Statement: {statement}\n"
        "Opinion:"
    ),
    "agree": (
        "Do you agree or disagree with the following statement?\n"
        "Statement: {statement}\n"
        "Answer:"
    ),
    "agree_yesno": (
        "Do you agree with the following statement?\n"
        "Statement: {statement}\n"
        "Answer (yes or no):"
    ),
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction: {statement}\n"
        "### Response:"
    ),
    "bare": "{statement}",
    "chat_qwen": (
        "<|im_start|>user\n{statement}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "chat_mistral": "[INST] {statement} [/INST]",
}

DEFAULT_TEMPLATE = "opinion"


def get_prompt(statement: str, template: str = DEFAULT_TEMPLATE) -> str:
    if template not in TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown template '{template}'. Choose from: {sorted(TEMPLATE_REGISTRY.keys())}"
        )
    return TEMPLATE_REGISTRY[template].format(statement=statement)

