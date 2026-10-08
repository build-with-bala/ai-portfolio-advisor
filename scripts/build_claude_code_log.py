"""Generate docs/CLAUDE_CODE_LOG.md from a real Claude Code session transcript.

    python scripts/build_claude_code_log.py ~/.claude/projects/<project>/<session-id>.jsonl

The log lists every prompt the user typed and every action the agent took
(tool, one-line purpose, files written), with timestamps. Tool outputs,
file contents and credentials are never copied.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))
HOME = str(Path.home())
# Harness messages that arrive in the user channel but were not typed by the user.
NOT_TYPED = ("task-notification", "SYSTEM NOTIFICATION", "agent-message", "Subagent hand-back")


def when(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST)


def clean(text: str, limit: int = 220) -> str:
    text = re.sub(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>", "[pasted: assignment specification]", text, flags=re.S)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = text.replace(HOME, "~").replace("|", "/")
    text = re.sub(r"\bwtf\b[ !]*", "[…]", text, flags=re.I)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(name: str, args: dict) -> str:
    if name == "Bash":
        return clean(args.get("description") or "shell command")
    if name in {"Write", "Edit", "Read"}:
        return clean(str(args.get("file_path", "")).split("2.Code.nosync/")[-1])
    if name == "Agent":
        return "delegate to sub-agent: " + clean(args.get("description", ""))
    if name == "Skill":
        return "load skill: " + clean(str(args.get("skill", "")))
    if name == "SendMessage":
        return "message sub-agent: " + clean(args.get("summary", ""))
    if name == "TaskStop":
        return "stop a background task"
    if name == "ToolSearch":
        return "load additional tools"
    if "playwright" in name:
        return "browser check (Playwright)"
    return name


def main(path: Path) -> None:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]
    model, timeline, tools, files = None, [], collections.Counter(), collections.Counter()
    for e in events:
        if e.get("isSidechain") or not e.get("timestamp"):
            continue
        ts, msg = when(e["timestamp"]), e.get("message") or {}
        content = msg.get("content")
        if e.get("type") == "user" and not e.get("isMeta"):
            texts = [content] if isinstance(content, str) else [b.get("text", "") for b in content or [] if b.get("type") == "text"]
            for text in texts:
                text = clean(text, 400)
                if text and not text.startswith("[Request interrupted") and not any(k in text for k in NOT_TYPED):
                    timeline.append((ts, "prompt", text))
        elif e.get("type") == "queue-operation" and e.get("operation") == "enqueue" and isinstance(e.get("content"), str):
            text = clean(e["content"], 400)
            if text and not any(k in text for k in NOT_TYPED):
                timeline.append((ts, "prompt", text + " (sent while the agent was working)"))
        elif e.get("type") == "assistant":
            model = msg.get("model") or model
            for block in content or []:
                if block.get("type") == "tool_use":
                    name, args = block["name"], block.get("input") or {}
                    label = "Playwright" if "playwright" in name else name
                    tools[label] += 1
                    if name in {"Write", "Edit"}:
                        files[describe(name, args)] += 1
                    timeline.append((ts, label, describe(name, args)))

    seen, deduped = set(), []
    for item in sorted(timeline, key=lambda x: x[0]):
        key = (item[1], item[2]) if item[1] == "prompt" else None
        if key and key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    timeline = deduped

    sub_dir = path.with_suffix("") / "subagents"
    subagents = []
    for sub in sorted(sub_dir.glob("*.jsonl")) if sub_dir.exists() else []:
        calls, first, last, task = 0, None, None, ""
        for line in sub.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("timestamp"):
                first, last = first or e["timestamp"], e["timestamp"]
            content = (e.get("message") or {}).get("content")
            if e.get("type") == "user" and not task:
                task = clean(content if isinstance(content, str) else " ".join(b.get("text", "") for b in content or [] if b.get("type") == "text"), 110)
            if e.get("type") == "assistant":
                calls += sum(1 for b in content or [] if b.get("type") == "tool_use")
        if first:
            subagents.append((when(first), when(last), calls, task))

    subagents.sort()
    start, end = timeline[0][0], timeline[-1][0]
    prompts = [t for t in timeline if t[1] == "prompt"]
    out = [
        "# Claude Code Session Log",
        "",
        "Generated by `scripts/build_claude_code_log.py` from the session transcript that Claude Code saves locally. "
        "Prompts are the user's own words; each action line is the purpose the agent recorded for that tool call. "
        "Tool outputs and file contents are not included.",
        "",
        "| | |",
        "|---|---|",
        f"| Session | `{path.stem}` |",
        f"| Model | `{model}` |",
        f"| Date | {start:%d %B %Y}, {start:%H:%M} to {end:%H:%M} IST |",
        f"| User prompts | {len(prompts)} |",
        f"| Agent tool calls | {sum(tools.values())} ({', '.join(f'{k} {v}' for k, v in tools.most_common())}) |",
        f"| Sub-agents | {len(subagents)} |",
        "",
        "## Prompts",
        "",
        "| Time (IST) | Prompt |",
        "|---|---|",
        *[f"| {ts:%H:%M:%S} | {text} |" for ts, _, text in prompts],
        "",
        "## Sub-agents",
        "",
        "| Started | Ran for | Tool calls | Task (first line of its brief) |",
        "|---|---|---|---|",
        *[f"| {a:%H:%M} | {int((b - a).total_seconds() // 60)} min | {calls} | {task} |" for a, b, calls, task in subagents],
        "",
        "## Timeline",
        "",
    ]
    for ts, kind, text in timeline:
        if kind == "prompt":
            out += ["", f"### {ts:%H:%M} — \"{text}\"", ""]
        else:
            out.append(f"- `{ts:%H:%M:%S}` **{kind}**: {text}")
    out += ["", "## Files written by the agent", "", *[f"- `{name}`" + (f" ({n} edits)" if n > 1 else "") for name, n in sorted(files.items())], ""]
    target = ROOT / "docs" / "CLAUDE_CODE_LOG.md"
    target.write_text("\n".join(out))
    print(f"{target.relative_to(ROOT)}: {len(prompts)} prompts, {sum(tools.values())} tool calls, {len(subagents)} sub-agents")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
