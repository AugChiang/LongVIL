"""Prompt templates for the LongVIL agents.

Templates live in ``templates/*.txt`` and contain ``{PLACEHOLDER}`` fields that
are filled with ``str.format``. Default values come from ``shared/*.txt`` and
``examples/*.txt``; any of them can be overridden via keyword arguments.

Note: ``<<authorized_imports>>``, ``<<tool_descriptions>>`` and
``<<managed_agents_descriptions>>`` are *not* ours -- transformers' agent
fills them in later, so they are kept verbatim in the text.
Literal braces inside a template must be escaped as ``{{`` / ``}}``.
"""
from pathlib import Path

_DIR = Path(__file__).parent

_SHARED = {
    "TOOL_USE_FORMAT": "shared/tool_use_format.txt",
    "ROBOT_PRIMITIVES": "shared/robot_primitives.txt",
    "RELATION_MAPPING": "shared/relation_mapping.txt",
}

_EXAMPLES = {
    "Planagent_prompt": "examples/Planagent_examples.txt",
    "Planagent_prompt_noreflection": "examples/Planagent_noreflection_examples.txt",
    "Codeagent_prompt": "examples/Codeagent_examples.txt",
    "Codeagent_prompt_noreflection": "examples/Codeagent_noreflection_examples.txt",
}


def _read(rel_path: str) -> str:
    return (_DIR / rel_path).read_text(encoding="utf-8")


def load_prompt(name: str, **overrides: str) -> str:
    """Read ``templates/<name>.txt`` and fill its ``{PLACEHOLDER}`` fields."""
    values = {key: _read(path).rstrip("\n") for key, path in _SHARED.items()}
    if name in _EXAMPLES:
        values["EXAMPLES"] = _read(_EXAMPLES[name]).rstrip("\n")
    values.update(overrides)
    return _read(f"templates/{name}.txt").format(**values)
