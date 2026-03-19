"""
Evo-skill management commands: list / delete / export / compose.

Usage:
    python main.py list
    python main.py delete <skill_id>
    python main.py export <output_path.json>
    python main.py compose "对一个数组先排序再二分查找目标值"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, List, Optional

from ..client import EvoSkill

_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_GRAY  = "\033[90m"
_RESET = "\033[0m"


def cmd_list(sdk: EvoSkill, user_id: str) -> None:
    """Print all skills for the given user."""
    skills = sdk.list(user_id=user_id)
    if not skills:
        print(f"{_GRAY}No skills found for user '{user_id}'.{_RESET}")
        return

    print(f"\n{_CYAN}{_BOLD}Skills for '{user_id}' ({len(skills)} total):{_RESET}\n")
    for i, s in enumerate(skills, 1):
        stats = sdk.store.get_skill_usage_stats(user_id=user_id, skill_id=s.id)
        skill_stats = (stats.get("skills") or {}).get(s.id, {})
        retrieved = skill_stats.get("retrieved", 0)
        used = skill_stats.get("used", 0)

        print(f"  {_BOLD}{i}. {s.name}{_RESET}  v{s.version}")
        print(f"     ID      : {s.id}")
        print(f"     Tags    : {', '.join(s.tags) if s.tags else '-'}")
        print(f"     Usage   : retrieved={retrieved}, used={used}")
        print(f"     Desc    : {s.description[:100]}")
        print()


def cmd_delete(sdk: EvoSkill, user_id: str, skill_id: str) -> None:
    """Delete a skill by ID."""
    skill = sdk.get(skill_id)
    if skill is None:
        print(f"{_RED}Skill not found: {skill_id}{_RESET}")
        sys.exit(1)

    confirm = input(f"Delete '{skill.name}' ({skill_id[:8]})? [y/N] ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return

    ok = sdk.delete(skill_id)
    if ok:
        print(f"{_CYAN}Deleted: {skill.name}{_RESET}")
    else:
        print(f"{_RED}Failed to delete skill.{_RESET}")
        sys.exit(1)


def cmd_export(sdk: EvoSkill, user_id: str, output_path: str) -> None:
    """Export all skills to a JSON file."""
    skills = sdk.list(user_id=user_id)
    if not skills:
        print(f"{_GRAY}No skills to export.{_RESET}")
        return

    records = []
    for s in skills:
        d = asdict(s)
        d.pop("source", None)  # strip raw source messages to keep output clean
        records.append(d)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"{_CYAN}Exported {len(records)} skills → {output_path}{_RESET}")


def cmd_compose(sdk: EvoSkill, user_id: str, task: str, llm: Any) -> None:
    """
    Compose a solution for a task by combining scripts from relevant skills.

    Steps:
    1. Search for skills matching the task description
    2. Collect their scripts/  files from disk
    3. Ask LLM to compose a complete solution using those scripts
    4. Print the composed solution (and optionally save it)
    """
    print(f"\n{_CYAN}{_BOLD}Composing solution for:{_RESET} {task}\n")

    # 1. Retrieve relevant skills
    hits = sdk.search(task, user_id=user_id, limit=5)
    if not hits:
        print(f"{_GRAY}No matching skills found. Try chatting first to build your skill bank.{_RESET}")
        return

    print(f"{_CYAN}Matched skills:{_RESET}")
    for h in hits:
        print(f"  {_BOLD}{h.skill.name}{_RESET}  (score={h.score:.3f})")

    # 2. Collect scripts from disk
    skill_scripts: List[dict] = []
    for h in hits:
        skill = h.skill
        scripts = {
            path: content
            for path, content in (skill.files or {}).items()
            if path.startswith("scripts/") and path.endswith(".py")
        }
        if scripts:
            skill_scripts.append({
                "skill_name": skill.name,
                "description": skill.description,
                "scripts": scripts,
            })
        else:
            # No script file saved yet — include instructions as context
            skill_scripts.append({
                "skill_name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "scripts": {},
            })

    has_scripts = any(s["scripts"] for s in skill_scripts)

    # 3. Build compose prompt
    skills_block = ""
    for s in skill_scripts:
        skills_block += f"\n## Skill: {s['skill_name']}\n"
        skills_block += f"Description: {s['description']}\n"
        for path, code in s.get("scripts", {}).items():
            skills_block += f"\n### {path}\n```python\n{code}\n```\n"
        if not s.get("scripts") and s.get("instructions"):
            skills_block += f"\nInstructions:\n{s['instructions']}\n"

    system = (
        "You are a Python programming assistant that composes solutions by combining skill modules.\n"
        "Given a task and a set of available skill scripts, write a complete Python solution.\n\n"
        "Rules:\n"
        "- Import and CALL the functions from the provided scripts where possible.\n"
        "- If a skill has no script yet, implement it inline based on its description/instructions.\n"
        "- Write clean, runnable code with a if __name__ == '__main__': demo.\n"
        "- Add comments explaining which skill each part comes from.\n"
        "- Output ONLY the Python code, no extra explanation.\n"
    )
    user = f"Task: {task}\n\nAvailable Skills:\n{skills_block}"

    print(f"\n{_GREEN}{_BOLD}Composed Solution:{_RESET}\n")
    full_code = ""
    try:
        for chunk in llm.stream_complete(system=system, user=user, temperature=0.2):
            print(chunk, end="", flush=True)
            full_code += chunk
    except Exception as e:
        print(f"\n{_RED}[LLM error: {e}]{_RESET}")
        return
    print()

    # 4. Offer to save
    save = input(f"\n{_GRAY}Save to file? (enter filename or press Enter to skip): {_RESET}").strip()
    if save:
        path = save if save.endswith(".py") else save + ".py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_code)
        print(f"{_CYAN}Saved → {path}{_RESET}")
