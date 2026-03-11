#!/usr/bin/env python3
"""
Mini-Castor Interactive Demo
=============================

Run this to see the kernel in action:

    python demo.py

What happens:
  1. A "research assistant" agent searches for emails, analyzes them,
     then tries to delete old ones.
  2. The kernel auto-suspends at the destructive "delete_emails" call.
  3. You choose: approve, reject, or modify.
  4. The kernel replays from checkpoint — watch it fast-forward through
     cached syscalls and continue live.

Two agent styles are demonstrated:
  - Classic style:  async def agent(proxy) — explicit proxy parameter
  - New style:      async def agent() — uses call_tool()/budget() via ContextVar

No API keys needed. Everything uses mock tools.
"""

from __future__ import annotations

import asyncio

from mini_castor import Kernel, SyscallProxy, budget, call_tool, tool


# ============================================================================
# MOCK TOOLS — These simulate real tools. No external dependencies.
# ============================================================================


@tool("io", cost=1)
async def search_emails(query: str) -> str:
    """Search the inbox. Safe, non-destructive."""
    return f"Found 847 emails matching '{query}'"


@tool("io", cost=1)
async def analyze(data: str) -> str:
    """Analyze data. Safe, non-destructive."""
    return f"Analysis: 712 of these emails are older than 30 days"


@tool("io", cost=2, destructive=True)
async def delete_emails(criteria: str) -> str:
    """Delete emails. DESTRUCTIVE — will trigger HITL suspension."""
    return f"Deleted emails matching: {criteria}"


@tool("io", cost=1)
async def send_summary(message: str) -> str:
    """Send a summary. Safe."""
    return f"Summary sent: {message}"


# ============================================================================
# AGENT FUNCTIONS — Two styles, same behavior
#
# Style 1 (classic): receives proxy explicitly — maximum control
# Style 2 (new):     uses call_tool()/budget() — zero kernel coupling
#
# Both work identically. The kernel auto-detects which style to use
# by inspecting the function signature (0 params = new, 1+ = classic).
# ============================================================================


async def research_assistant_classic(proxy: SyscallProxy) -> str:
    """Classic style: explicit proxy parameter.

    This is what runs in "user space". It ONLY interacts with the kernel
    through proxy.syscall(). It has no idea about replay, budgets, or HITL —
    the kernel handles all of that transparently.
    """

    # Step 1: Search (fast path — safe, has budget)
    emails = await proxy.syscall("search_emails", {"query": "older than 30 days"})
    print(f"    Agent received: {emails}")

    # Step 2: Analyze (fast path)
    analysis = await proxy.syscall("analyze", {"data": emails})
    print(f"    Agent received: {analysis}")

    # Step 3: Delete (SLOW PATH — destructive!)
    # This will suspend the agent for human review.
    # On replay after approval, the cached result is returned instantly.
    result = await proxy.syscall("delete_emails", {"criteria": "older than 30 days"})
    print(f"    Agent received: {result}")

    # If we got a rejection or modification, the "LLM" adapts
    if isinstance(result, dict) and result.get("status") == "REJECTED":
        print(f"    Agent sees rejection: {result['feedback']}")
        summary = await proxy.syscall(
            "send_summary", {"message": f"Action blocked: {result['feedback']}"}
        )
    elif isinstance(result, dict) and result.get("status") == "MODIFIED":
        print(f"    Agent sees modification: {result['feedback']}")
        result2 = await proxy.syscall(
            "delete_emails", {"criteria": "older than 90 days"}
        )
        print(f"    Agent received (revised): {result2}")
        summary = await proxy.syscall(
            "send_summary", {"message": f"Cleaned up with modification: {result2}"}
        )
    else:
        summary = await proxy.syscall(
            "send_summary", {"message": f"Cleanup complete: {result}"}
        )

    print(f"    Agent received: {summary}")
    return "Done"


async def research_assistant_new() -> str:
    """New style: uses call_tool()/budget() — no proxy parameter.

    This agent has ZERO kernel imports in its logic. It only uses call_tool()
    and budget(), which find the proxy via a ContextVar set by the kernel.

    In real Castor, these come from `castor.lib` — a separate package that
    agent developers import. Operators never touch castor.lib; agents never
    touch castor.core. Clean separation.
    """

    # Step 1: Search (fast path)
    emails = await call_tool("search_emails", query="older than 30 days")
    print(f"    Agent received: {emails}")

    # Check budget — no proxy needed!
    remaining = budget("io")
    print(f"    Budget remaining: {remaining}")

    # Step 2: Analyze (fast path)
    analysis = await call_tool("analyze", data=emails)
    print(f"    Agent received: {analysis}")

    # Step 3: Delete (SLOW PATH — destructive!)
    result = await call_tool("delete_emails", criteria="older than 30 days")
    print(f"    Agent received: {result}")

    if isinstance(result, dict) and result.get("status") == "REJECTED":
        print(f"    Agent sees rejection: {result['feedback']}")
        summary = await call_tool(
            "send_summary", message=f"Action blocked: {result['feedback']}"
        )
    elif isinstance(result, dict) and result.get("status") == "MODIFIED":
        print(f"    Agent sees modification: {result['feedback']}")
        result2 = await call_tool("delete_emails", criteria="older than 90 days")
        print(f"    Agent received (revised): {result2}")
        summary = await call_tool(
            "send_summary", message=f"Cleaned up with modification: {result2}"
        )
    else:
        summary = await call_tool(
            "send_summary", message=f"Cleanup complete: {result}"
        )

    print(f"    Agent received: {summary}")
    return "Done"


# ============================================================================
# DEMO RUNNER
# ============================================================================

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text: str):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{RESET}\n")


def print_checkpoint(cp):
    print(f"  {YELLOW}PID:{RESET}    {cp.pid}")
    print(f"  {YELLOW}Status:{RESET} {cp.status}")
    print(f"  {YELLOW}Budget:{RESET} {cp.budgets}")
    print(f"  {YELLOW}Log:{RESET}    {len(cp.syscall_log)} syscalls recorded")
    if cp.pending_hitl:
        print(f"  {RED}PENDING HITL:{RESET} {cp.pending_hitl['tool_name']}"
              f"({cp.pending_hitl['arguments']})")
    if cp.result is not None:
        print(f"  {GREEN}Result:{RESET} {cp.result}")


async def main():
    print_header("Mini-Castor Demo: The xv6 of Agent Operating Systems")

    print(f"  This demo shows an agent that searches, analyzes, and tries to")
    print(f"  delete emails. The kernel auto-suspends at the destructive call.")
    print(f"  You decide what happens next.\n")

    # ── Choose agent style ──
    print(f"  {BOLD}Agent style:{RESET}")
    print(f"    {GREEN}[1]{RESET} Classic  — async def agent(proxy) — explicit proxy")
    print(f"    {GREEN}[2]{RESET} New      — async def agent() — uses call_tool()/budget()")
    print()

    style = input(f"  {BOLD}Choose style (1/2, default=2): {RESET}").strip()
    if style == "1":
        agent_name = "classic"
        agent_fn = research_assistant_classic
        print(f"\n  Using {CYAN}classic style{RESET} (explicit proxy parameter)\n")
    else:
        agent_name = "new"
        agent_fn = research_assistant_new
        print(f"\n  Using {CYAN}new style{RESET} (call_tool/budget via ContextVar)\n")

    # ── Phase 1: Run agent until it suspends ──
    print_header("Phase 1: Running agent (will suspend at delete_emails)")

    kernel = Kernel(budgets={"io": 20})
    kernel.register_agent(agent_name, agent_fn)

    checkpoint = await kernel.run(agent_name)

    print(f"\n  {BOLD}Agent suspended!{RESET} Here's the checkpoint:\n")
    print_checkpoint(checkpoint)

    # ── Phase 2: Human decides ──
    print_header("Phase 2: Human-in-the-Loop Decision")

    print(f"  The agent wants to: {RED}delete_emails{RESET}"
          f"(criteria='older than 30 days')\n")
    print(f"  Choose your response:")
    print(f"    {GREEN}[a]{RESET} Approve  — execute the delete as requested")
    print(f"    {RED}[r]{RESET} Reject   — block the action, agent re-plans")
    print(f"    {YELLOW}[m]{RESET} Modify   — approve intent, but change scope")
    print()

    choice = input(f"  {BOLD}Your choice (a/r/m): {RESET}").strip().lower()

    if choice == "a":
        print(f"\n  {GREEN}Approving...{RESET}")
        await kernel.approve(checkpoint)
    elif choice == "r":
        feedback = input(f"  {BOLD}Rejection reason: {RESET}") or "Too risky"
        print(f"\n  {RED}Rejecting...{RESET}")
        kernel.reject(checkpoint, feedback)
    elif choice == "m":
        feedback = input(
            f"  {BOLD}Modification feedback: {RESET}"
        ) or "Only delete emails older than 90 days"
        print(f"\n  {YELLOW}Modifying...{RESET}")
        kernel.modify(checkpoint, feedback)
    else:
        print(f"\n  Unknown choice, defaulting to reject.")
        kernel.reject(checkpoint, "Unknown input")

    print(f"\n  Checkpoint after HITL resolution:")
    print_checkpoint(checkpoint)

    # ── Phase 3: Resume via replay ──
    print_header("Phase 3: Resuming via Replay")

    print(f"  The kernel will re-run the agent function from the TOP.")
    print(f"  Watch: the first {len(checkpoint.syscall_log)} syscalls are served")
    print(f"  from cache (replay). Then execution continues live.\n")

    checkpoint = await kernel.run(agent_name, checkpoint)

    print(f"\n  {BOLD}Agent completed!{RESET} Final checkpoint:\n")
    print_checkpoint(checkpoint)

    # ── Show the full audit trail ──
    print_header("Audit Trail: Full Syscall Log")

    for i, record in enumerate(checkpoint.syscall_log):
        hitl_marker = f" {YELLOW}[HITL]{RESET}" if record.was_hitl else ""
        tool_name = record.request["tool_name"]
        response_preview = str(record.response)[:60]
        print(f"  #{i}: {tool_name}{hitl_marker} -> {response_preview}")

    print(f"\n  {GREEN}Total syscalls: {len(checkpoint.syscall_log)}{RESET}")
    print(f"  {GREEN}Budget remaining: {checkpoint.budgets}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
