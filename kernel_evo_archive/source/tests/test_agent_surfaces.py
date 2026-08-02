from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OPENCODE_AGENTS = (
    "kernelevo-island-author",
    "kernelevo-profile-reviewer",
    "kernelevo-repair-author",
)


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, raw, body = text.split("---\n", 2)
    payload = yaml.safe_load(raw)
    assert isinstance(payload, dict)
    return payload, body


def test_opencode_agents_inherit_selected_model_and_are_bounded() -> None:
    for name in OPENCODE_AGENTS:
        metadata, body = _frontmatter(ROOT / ".opencode" / "agents" / f"{name}.md")
        assert metadata["mode"] == "all"
        assert "model" not in metadata
        assert metadata["permission"]["read"] == "allow"
        assert metadata["permission"]["edit"]["*"] == "deny"
        for permission in ("bash", "task", "webfetch", "websearch", "skill"):
            assert metadata["permission"][permission] == "deny"
        assert "Do not" in body
        assert "run" in body.lower()

    author, _ = _frontmatter(
        ROOT / ".opencode" / "agents" / "kernelevo-island-author.md"
    )
    repair, _ = _frontmatter(
        ROOT / ".opencode" / "agents" / "kernelevo-repair-author.md"
    )
    reviewer, _ = _frontmatter(
        ROOT / ".opencode" / "agents" / "kernelevo-profile-reviewer.md"
    )
    candidate_pattern = "**/iter_*/island_*/candidate/*.py"
    assert author["permission"]["edit"][candidate_pattern] == "allow"
    assert repair["permission"]["edit"][candidate_pattern] == "allow"
    assert (
        reviewer["permission"]["edit"][
            "**/iter_*/island_*/context/PROFILE_REVIEW.json"
        ]
        == "allow"
    )


def test_all_agent_surfaces_cover_profile_review_and_repair() -> None:
    codex_skill = (ROOT / ".agents" / "skills" / "kernelevo" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    claude_skill = (ROOT / ".claude" / "skills" / "kernelevo" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    for skill in (codex_skill, claude_skill):
        assert "iter review-profiles" in skill
        assert "island review-submit" in skill
        assert "island repair" in skill
    assert "Codex or OpenCode" in codex_skill

    reviewer, reviewer_body = _frontmatter(
        ROOT / ".claude" / "agents" / "kernelevo-profile-reviewer.md"
    )
    assert reviewer["model"] == "inherit"
    assert "Write" in reviewer["tools"]
    assert "PROFILE_REVIEW.json" in reviewer_body


def test_opencode_ollama_example_has_no_secret_and_uses_custom_provider() -> None:
    config_path = ROOT / "examples" / "agent" / "opencode-ollama.json"
    raw = config_path.read_text(encoding="utf-8")
    config = json.loads(raw)
    provider = config["provider"]["ollama"]

    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:11434/v1"
    assert "gpt-oss:20b" in provider["models"]
    assert "qwen3.5" in provider["models"]
    assert "apiKey" not in raw
