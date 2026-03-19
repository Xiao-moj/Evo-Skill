"""
Evo-skill interactive chat — terminal chat loop with automatic skill retrieval and extraction.

Usage:
    python main.py chat
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..agent_runtime import AgentContext, run_agent_session
from ..client import EvoSkill
from ..interactive.capability_analyzer import LLMCapabilityAnalyzer
from ..interactive.commands import parse_command
from ..interactive.config import InteractiveConfig
from ..interactive.gating import heuristic_is_ack_feedback, heuristic_topic_changed
from ..interactive.retrieval import retrieve_hits_by_scope, capability_pre_recall
from ..interactive.rewriting import LLMQueryRewriter
from ..interactive.selection import LLMSkillSelector
from ..interactive.usage_tracking import LLMSkillUsageJudge
from ..management.agent_failure_extraction import AgentFailureExperienceExtractor
from ..management.agent_trajectory_extraction import AgentTrajectoryExtractor
from ..management.reviewer import LLMSkillReviewer
from ..render import render_experience_context, render_skills_context


# ── ANSI colors ───────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_GRAY   = "\033[90m"
_RED    = "\033[31m"


def _print_skill_hits(hits: list) -> None:
    if not hits:
        return
    print(f"\n{_CYAN}[Skills retrieved]{_RESET}")
    for h in hits:
        s = h.skill
        print(f"  {_BOLD}{s.name}{_RESET} v{s.version}  score={h.score:.3f}")


def _build_generation_payload(
    messages: List[Dict[str, Any]],
    skill_context: str,
) -> Tuple[str, str]:
    """Builds the system/user payload used by the generation backend."""

    system = (
        "You are a helpful assistant."
        + (f"\n\n{skill_context}" if skill_context else "")
    )
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return system, ""
    last_user = user_msgs[-1]["content"]

    history_for_llm: List[Dict[str, str]] = []
    for m in messages[:-1]:
        if m.get("role") in ("user", "assistant"):
            history_for_llm.append({"role": m["role"], "content": str(m.get("content", ""))})

    if history_for_llm:
        history_text = "\n".join(
            f"[{m['role']}] {m['content']}" for m in history_for_llm[-12:]
        )
        full_user = f"Conversation history:\n{history_text}\n\n[user] {last_user}"
    else:
        full_user = last_user
    return system, full_user


def _stream_assistant(
    llm,
    *,
    system: str,
    user: str,
    temperature: float,
) -> str:
    """Streams one assistant reply from the built-in LLM backend."""

    if not user:
        return ""
    print(f"\n{_GREEN}Assistant:{_RESET} ", end="", flush=True)
    full_reply = ""
    try:
        for chunk in llm.stream_complete(system=system, user=user, temperature=temperature):
            print(chunk, end="", flush=True)
            full_reply += chunk
    except Exception as e:
        print(f"\n{_RED}[LLM error: {e}]{_RESET}")
    print()
    return full_reply


def _generate_assistant_reply(
    *,
    sdk: EvoSkill,
    llm,
    cfg: InteractiveConfig,
    messages: List[Dict[str, Any]],
    selected_skills: List[Any],
    skill_context: str,
    experience_context: str,
) -> Tuple[str, bool, Optional[AgentContext]]:
    """Generates one assistant reply using the configured backend."""

    llm_skill_context = skill_context if cfg.agent_backend == "llm" else ""
    llm_experience_context = experience_context if cfg.agent_backend == "llm" else ""
    system, user = _build_generation_payload(messages, llm_skill_context)
    if llm_experience_context:
        system = f"{system}\n\n{llm_experience_context}".strip()
    if cfg.agent_backend == "llm":
        reply = _stream_assistant(
            llm,
            system=system,
            user=user,
            temperature=cfg.assistant_temperature,
        )
        return reply, bool(reply), None

    runtime_system = "You are a helpful assistant."
    if selected_skills:
        runtime_system += "\nRelevant skills may be available in the runtime. Use them when helpful."
    instruction_parts = [f"System instruction:\n{runtime_system}"]
    if experience_context:
        instruction_parts.append(experience_context)
    if user:
        instruction_parts.append(user)
    instruction = "\n\n".join(part for part in instruction_parts if str(part or "").strip()).strip()

    print(f"\n{_GREEN}Assistant:{_RESET} ", end="", flush=True)
    context = run_agent_session(
        sdk=sdk,
        cfg=cfg,
        instruction=instruction,
        selected_skills=selected_skills,
    )
    reply = str(context.response_content or "").strip()
    if reply:
        print(reply, end="", flush=True)
    elif context.success is False:
        print(f"[{cfg.agent_backend} returned no response]", end="", flush=True)
    print()
    if context.success is False:
        logs_dir = str((context.metadata or {}).get("logs_dir") or "").strip()
        if logs_dir:
            print(f"{_GRAY}[Agent logs] {logs_dir}{_RESET}")
    return reply, bool(context.success), context


def _top_reference_from_hits(hits: list, *, user_id: str) -> Optional[Dict[str, Any]]:
    """Return the top-1 hit as a dict for extraction context."""
    for h in hits:
        s = h.skill
        if str(getattr(s, "user_id", "") or "") == user_id:
            return {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "triggers": list(s.triggers or [])[:6],
            }
    if hits:
        s = hits[0].skill
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "triggers": list(s.triggers or [])[:6],
        }
    return None


def run_chat(
    sdk: EvoSkill,
    llm,
    cfg: InteractiveConfig,
) -> None:
    """Main interactive chat loop."""

    rewriter = LLMQueryRewriter(
        llm=llm,
        max_history_turns=cfg.rewrite_history_turns,
        max_history_chars=cfg.rewrite_history_chars,
        max_query_chars=cfg.rewrite_max_query_chars,
    )
    capability_analyzer = LLMCapabilityAnalyzer(llm=llm)
    selector = LLMSkillSelector(llm=llm)
    reviewer = LLMSkillReviewer(llm=llm)
    trajectory_extractor = AgentTrajectoryExtractor(sdk.config, llm=llm)
    failure_extractor = AgentFailureExperienceExtractor(sdk.config, llm=llm)
    usage_judge = LLMSkillUsageJudge(llm=llm) if cfg.usage_tracking_enabled else None

    bg_queue: queue.Queue = queue.Queue()
    epoch = 0

    messages: List[Dict[str, Any]] = []
    turn = 0
    pending_extraction: Optional[Dict[str, Any]] = None
    last_user_msg: str = ""
    last_assistant_msg: str = ""

    _print_banner(cfg)

    def _review_upserted_skills(updated: List[Any], item: Dict[str, Any]) -> None:
        """Runs reviewer checks and follow-up retries for extracted skills."""

        if not updated:
            return
        for skill in updated:
            result = reviewer.review(skill)
            label = "merge" if result.is_merge else "new"
            if not result.approved:
                if result.is_merge:
                    retry_count = item.get("retry_count", 0)
                    if retry_count >= 1:
                        print(
                            f"\n{_YELLOW}[Skill merge warning: {skill.name}]{_RESET} "
                            f"{_GRAY}Retry limit reached. {result.reason}{_RESET}",
                            flush=True,
                        )
                    else:
                        accumulated_skips = set(item.get("skip_merge_ids") or set()) | {skill.id}
                        print(
                            f"\n{_YELLOW}[Skill merge rejected: {skill.name}]{_RESET} "
                            f"{_GRAY}Retrying as new skill...{_RESET}",
                            flush=True,
                        )
                        retry_item = {
                            "kind": item.get("kind", "extract"),
                            "epoch": epoch,
                            "hint": item.get("hint"),
                            "retrieval_reference": item.get("retrieval_reference"),
                            "skip_merge_ids": accumulated_skips,
                            "retry_count": retry_count + 1,
                        }
                        if item.get("kind") == "extract_trace":
                            retry_item["context"] = item.get("context")
                        else:
                            retry_item["window"] = item.get("window")
                        bg_queue.put(retry_item)
                else:
                    sdk.delete(skill.id)
                    print(
                        f"\n{_RED}[Skill rejected (new): {skill.name}]{_RESET} "
                        f"{_GRAY}{result.reason}{_RESET}",
                        flush=True,
                    )
            else:
                flag = ""
                if result.is_merge and result.regression:
                    flag = f" {_YELLOW}⚠ regression: {result.reason}{_RESET}"
                elif result.is_merge:
                    flag = f" {_GRAY}(merged, score={result.score:.2f}){_RESET}"
                print(
                    f"\n{_YELLOW}[Skill saved ({label}): {skill.name}]{_RESET}"
                    f"{flag}",
                    flush=True,
                )

    def _bg_worker() -> None:
        while True:
            item = bg_queue.get()
            if item is None:
                break
            kind = item.get("kind")
            try:
                if kind == "extract":
                    if item.get("epoch") != epoch:
                        continue
                    print(f"\n{_GRAY}[Extracting skills...]{_RESET}", flush=True)
                    updated = sdk.ingest(
                        messages=item["window"],
                        user_id=cfg.user_id,
                        hint=item.get("hint"),
                        metadata={"extraction_reference": item.get("retrieval_reference")},
                        skip_merge_ids=item.get("skip_merge_ids"),
                    )
                    _review_upserted_skills(updated, item)
                elif kind == "extract_trace":
                    if item.get("epoch") != epoch:
                        continue
                    context = item.get("context")
                    if not isinstance(context, AgentContext):
                        continue
                    print(f"\n{_GRAY}[Extracting skills from trace...]{_RESET}", flush=True)
                    candidates = trajectory_extractor.extract_from_context(
                        user_id=cfg.user_id,
                        context=context,
                        max_candidates=max(0, int(sdk.config.max_candidates_per_ingest)),
                        hint=item.get("hint"),
                        retrieved_reference=item.get("retrieval_reference"),
                    )
                    updated = sdk.maintainer.apply(
                        candidates,
                        user_id=cfg.user_id,
                        metadata={"extraction_reference": item.get("retrieval_reference")},
                        skip_ids=item.get("skip_merge_ids"),
                    )
                    _review_upserted_skills(updated, item)
                elif kind == "extract_experience":
                    if item.get("epoch") != epoch:
                        continue
                    context = item.get("context")
                    if not isinstance(context, AgentContext):
                        continue
                    print(f"\n{_GRAY}[Extracting failure experience...]{_RESET}", flush=True)
                    candidates = failure_extractor.extract_from_context(
                        user_id=cfg.user_id,
                        context=context,
                        max_candidates=max(1, min(3, int(sdk.config.max_candidates_per_ingest or 1))),
                        hint=item.get("hint"),
                    )
                    updated = sdk.maintainer.apply(
                        candidates,
                        user_id=cfg.user_id,
                        metadata=None,
                        skip_ids=item.get("skip_merge_ids"),
                    )
                    for experience in updated:
                        print(
                            f"\n{_YELLOW}[Experience saved: {experience.name}]{_RESET}",
                            flush=True,
                        )
                elif kind == "usage":
                    if item.get("epoch") != epoch or usage_judge is None:
                        continue
                    judgments = usage_judge.judge(
                        query=item["query"],
                        assistant_reply=item["reply"],
                        hits=item["hits"],
                        selected_for_context_ids=[s.id for s in item["selected"]],
                    )
                    if judgments:
                        sdk.store.record_skill_usage_judgments(
                            user_id=cfg.user_id,
                            judgments=judgments,
                            prune_min_retrieved=cfg.usage_prune_min_retrieved,
                            prune_max_used=cfg.usage_prune_max_used,
                        )
            except Exception as e:
                print(f"\n{_GRAY}[bg error: {e}]{_RESET}", flush=True)
            finally:
                bg_queue.task_done()

    bg_thread = threading.Thread(target=_bg_worker, daemon=True)
    bg_thread.start()

    try:
        while True:
            try:
                raw = input(f"\n{_BOLD}You:{_RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_GRAY}Bye!{_RESET}")
                break

            if not raw:
                continue

            # ── Built-in commands ──────────────────────────────────────────
            cmd = parse_command(raw)
            if cmd:
                if cmd.name in {"/exit", "/quit"}:
                    print(f"{_GRAY}Bye!{_RESET}")
                    break
                elif cmd.name == "/help":
                    _print_help()
                elif cmd.name == "/skills":
                    _cmd_list_skills(sdk, cfg.user_id)
                elif cmd.name == "/compose":
                    if not cmd.arg:
                        print(f"{_GRAY}Usage: /compose <task description>{_RESET}")
                    else:
                        from ..cli.commands import cmd_compose
                        cmd_compose(sdk, cfg.user_id, cmd.arg, llm)
                elif cmd.name == "/clear":
                    epoch += 1
                    messages.clear()
                    pending_extraction = None
                    last_user_msg = ""
                    last_assistant_msg = ""
                    turn = 0
                    print(f"{_GRAY}[History cleared]{_RESET}")
                elif cmd.name == "/extract_now":
                    if messages:
                        bg_queue.put({
                            "kind": "extract",
                            "epoch": epoch,
                            "window": list(messages[-cfg.ingest_window:]),
                            "hint": cmd.arg or None,
                            "retrieval_reference": None,
                        })
                        print(f"{_GRAY}[Extraction queued]{_RESET}")
                    else:
                        print(f"{_GRAY}[No history to extract from]{_RESET}")
                else:
                    print(f"{_GRAY}Unknown command: {cmd.name}. Type /help{_RESET}")
                continue

            # ── Retrieval ─────────────────────────────────────────────────
            query = raw
            if cfg.rewrite_mode in {"always", "auto"} and messages:
                query = rewriter.rewrite(query=raw, messages=messages)
                if query != raw:
                    print(f"{_GRAY}[Query rewritten] {query}{_RESET}")

            # Capability decomposition → pre-recall
            capabilities = capability_analyzer.analyze(query)
            if capabilities:
                print(f"{_GRAY}[Capabilities] {', '.join(capabilities)}{_RESET}")

            retrieval = retrieve_hits_by_scope(
                sdk=sdk,
                query=query,
                user_id=cfg.user_id,
                scope=cfg.skill_scope,
                top_k=cfg.top_k,
                min_score=cfg.min_score,
            )
            hits = retrieval["hits"]

            # Capability pre-recall: find skills whose capabilities overlap with required ones
            if capabilities:
                pre_hits = capability_pre_recall(
                    sdk=sdk,
                    user_id=cfg.user_id,
                    capabilities=capabilities,
                    existing_hit_ids={h.skill.id for h in hits},
                )
                if pre_hits:
                    names = ", ".join(h.skill.name for h in pre_hits)
                    print(f"{_GRAY}[Capability pre-recall] {names}{_RESET}")
                    hits = list(hits) + pre_hits

            if not hits:
                print(f"{_GRAY}[No matching skills]{_RESET}")

            # LLM skill selection
            selected_skills = []
            if hits:
                candidate_skills = [h.skill for h in hits]
                selected_skills = selector.select(
                    query=query,
                    messages=messages,
                    skills=candidate_skills,
                )
                if selected_skills:
                    names = ", ".join(s.name for s in selected_skills)
                    print(f"{_GRAY}[Skills selected for context] {names}{_RESET}")
                else:
                    print(f"{_GRAY}[Skills retrieved but none selected]{_RESET}")

            if hits:
                _print_skill_hits(hits)

            # Recall failure experiences / guardrails for Docker-backed agents.
            selected_experiences = []
            experience_context = ""
            if cfg.agent_backend != "llm":
                try:
                    experience_hits = sdk.search_experiences(
                        query,
                        user_id=cfg.user_id,
                        limit=max(1, min(3, int(cfg.top_k or 3))),
                        scope="user",
                    )
                except Exception:
                    experience_hits = []
                experience_hits = [
                    hit for hit in experience_hits if float(getattr(hit, "score", 0.0) or 0.0) >= cfg.min_score
                ]
                selected_experiences = [hit.skill for hit in experience_hits[:3]]
                if selected_experiences:
                    names = ", ".join(s.name for s in selected_experiences)
                    print(f"{_GRAY}[Failure guardrails recalled] {names}{_RESET}")
                    experience_context = render_experience_context(
                        selected_experiences,
                        query=query,
                        max_chars=max(1200, min(3000, sdk.config.max_context_chars // 2)),
                    )

            # Render context
            skill_context = ""
            if selected_skills and cfg.agent_backend == "llm":
                skill_context = render_skills_context(
                    selected_skills,
                    query=query,
                    max_chars=sdk.config.max_context_chars,
                )

            # ── Add user turn ─────────────────────────────────────────────
            messages.append({"role": "user", "content": raw})

            # ── Topic change detection ────────────────────────────────────
            # If topic changed, only pass the current message to the LLM.
            # Full messages list is kept intact for skill extraction.
            topic_changed = (
                turn > 0
                and last_user_msg
                and heuristic_topic_changed(last_user_msg, last_assistant_msg, raw)
            )
            if topic_changed:
                print(f"{_GRAY}[Topic change detected — fresh context]{_RESET}")
                llm_messages = [{"role": "user", "content": raw}]
            else:
                llm_messages = messages

            # ── Generate reply ────────────────────────────────────────────
            reply, reply_ok, agent_context = _generate_assistant_reply(
                sdk=sdk,
                llm=llm,
                cfg=cfg,
                messages=llm_messages,
                selected_skills=selected_skills,
                skill_context=skill_context,
                experience_context=experience_context,
            )
            if reply and reply_ok:
                messages.append({"role": "assistant", "content": reply})

            clean_reply = reply if reply_ok else ""
            last_user_msg = raw
            last_assistant_msg = clean_reply
            turn += 1

            # ── Background: usage tracking ────────────────────────────────
            if hits and reply and reply_ok and usage_judge is not None:
                bg_queue.put({
                    "kind": "usage",
                    "epoch": epoch,
                    "query": query,
                    "reply": reply,
                    "hits": list(hits),
                    "selected": list(selected_skills),
                })

            # ── Extraction gating ─────────────────────────────────────────
            if cfg.extract_mode == "never":
                pass
            elif turn % cfg.extract_turn_limit == 0:
                window = list(messages[-cfg.ingest_window:])
                ref = _top_reference_from_hits(hits, user_id=cfg.user_id)
                if (
                    cfg.agent_backend != "llm"
                    and isinstance(agent_context, AgentContext)
                    and (agent_context.trajectory or agent_context.success is False)
                ):
                    if not heuristic_is_ack_feedback(raw):
                        if agent_context.success:
                            bg_queue.put({
                                "kind": "extract_trace",
                                "epoch": epoch,
                                "context": agent_context,
                                "hint": None,
                                "retrieval_reference": ref,
                            })
                        else:
                            bg_queue.put({
                                "kind": "extract_experience",
                                "epoch": epoch,
                                "context": agent_context,
                                "hint": None,
                            })
                    continue
                # Check if previous pending extraction should fire (topic changed)
                if pending_extraction and heuristic_topic_changed(
                    pending_extraction.get("latest_user", ""),
                    pending_extraction.get("latest_assistant", ""),
                    raw,
                ):
                    bg_queue.put({
                        "kind": "extract",
                        "epoch": epoch,
                        "window": pending_extraction["window"],
                        "hint": None,
                        "retrieval_reference": pending_extraction.get("ref"),
                    })
                    pending_extraction = None

                if not heuristic_is_ack_feedback(raw):
                    pending_extraction = {
                        "latest_user": raw,
                        "latest_assistant": clean_reply,
                        "window": window,
                        "ref": ref,
                    }
                    bg_queue.put({
                        "kind": "extract",
                        "epoch": epoch,
                        "window": window,
                        "hint": None,
                        "retrieval_reference": ref,
                    })

    finally:
        bg_queue.put(None)
        bg_thread.join(timeout=5)


def _cmd_list_skills(sdk: EvoSkill, user_id: str) -> None:
    skills = sdk.list(user_id=user_id)
    if not skills:
        print(f"{_GRAY}[No skills yet]{_RESET}")
        return
    print(f"\n{_CYAN}Your skills ({len(skills)}):{_RESET}")
    for s in skills:
        print(f"  {_BOLD}{s.name}{_RESET}  v{s.version}  [{s.id[:8]}]")
        print(f"    {_GRAY}{s.description[:80]}{_RESET}")


def _print_help() -> None:
    print(f"""
{_BOLD}Evo-skill commands:{_RESET}
  /help                   Show this help
  /skills                 List all your skills
  /compose <task>         Compose a solution from saved skill scripts
  /extract_now [hint]     Force skill extraction from current history
  /clear                  Clear conversation history
  /exit, /quit            Exit
""")


def _print_banner(cfg: InteractiveConfig) -> None:
    print(f"""
{_BOLD}{_CYAN}╔═══════════════════════════════╗
║        Evo-skill Chat         ║
╚═══════════════════════════════╝{_RESET}
User: {cfg.user_id}  |  Scope: {cfg.skill_scope}  |  Type /help for commands
Backend: {cfg.agent_backend}
""")
