"""
Mini-Castor: The xv6 of Agent Operating Systems
=================================================

The ENTIRE agent microkernel in one file (~500 lines with comments).

This file teaches six concepts from real OS design, applied to LLM agents:

  1. SYSCALL PROXY    — All agent actions go through a single gateway (like
                        Linux syscalls). The proxy decides: replay from cache,
                        execute live, or suspend for human review.

  2. CHECKPOINT/REPLAY — Agent state is a log of completed syscalls. To resume
                         a suspended agent, we re-run the function from the top
                         and serve cached responses. The agent doesn't know it's
                         being replayed. (We can't pickle Python coroutines, so
                         this is how we "serialize" async execution state.)

  3. CAPABILITY BUDGET — Every tool costs something. Agents get a finite budget.
                         Children can never exceed their parent's allocation.
                         This prevents runaway LLMs from making unlimited API calls.

  4. HITL FEEDBACK     — Destructive tools auto-suspend. Humans can approve,
                         reject, or modify. "Modify" is the key insight: we log
                         the human's feedback and let the LLM re-plan on replay,
                         rather than mutating the blocked request (which would
                         break replay determinism).

  5. CONTEXTVAR BRIDGE — Agents can be written WITHOUT a proxy parameter.
                         A ContextVar holds the current proxy; free functions
                         like call_tool() look it up implicitly. The kernel sets
                         it before running the agent. This separates "operator"
                         (kernel setup) from "agent developer" (pure logic).

  6. DUAL SIGNATURE    — The kernel auto-detects the agent's signature.
                         `async def agent(proxy)` = classic style (explicit).
                         `async def agent()` = new style (implicit via ContextVar).
                         Both are fully supported. Agents migrate at their own pace.

Usage (classic — explicit proxy):

    from mini_castor import Kernel, tool

    @tool("io", destructive=True)
    async def delete_file(path: str) -> str: ...

    async def my_agent(proxy):
        result = await proxy.syscall("read_file", {"path": "x.txt"})
        await proxy.syscall("delete_file", {"path": "x.txt"})  # suspends!
        return result

    kernel = Kernel(budgets={"io": 10})
    kernel.register_agent("my_agent", my_agent)
    checkpoint = await kernel.run("my_agent")

Usage (new — implicit via ContextVar):

    from mini_castor import Kernel, tool, call_tool, budget

    @tool("io")
    async def search(query: str) -> str: ...

    async def my_agent():
        result = await call_tool("search", query="hello")
        remaining = budget("io")
        return f"Found: {result} ({remaining} budget left)"

    kernel = Kernel(budgets={"io": 10})
    kernel.register_agent("my_agent", my_agent)
    checkpoint = await kernel.run("my_agent")
"""

from __future__ import annotations

import asyncio
import inspect
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# PART 1: DATA MODELS
#
# These are the "structs" of our micro-OS. In real Castor, these are Pydantic
# models. Here we use plain dataclasses for zero dependencies.
# ============================================================================


@dataclass
class SyscallRecord:
    """One completed syscall in the replay log.

    This is the fundamental unit of the checkpoint/replay model.
    The entire agent execution history is a list of these records.
    """

    request: dict[str, Any]  # {"tool_name": "read", "arguments": {"path": "x"}}
    response: Any  # Whatever the tool returned
    was_hitl: bool = False  # True if this went through human review


@dataclass
class Checkpoint:
    """The complete, serializable state of an agent process.

    This IS the process. There's no coroutine frame, no closure, no stack.
    Just a log of what happened and what's pending.

    To "resume" an agent: re-run the function from the top with this checkpoint.
    The SyscallProxy serves cached responses from syscall_log, fast-forwarding
    to where the agent left off. Then execution continues live.
    """

    pid: str
    status: str = "RUNNING"  # RUNNING | SUSPENDED | COMPLETED | PREEMPTED
    syscall_log: list[SyscallRecord] = field(default_factory=list)
    pending_hitl: dict[str, Any] | None = None  # The blocked syscall request
    budgets: dict[str, float] = field(default_factory=dict)  # {resource: remaining}
    result: Any = None


class SuspendInterrupt(Exception):
    """Raised to unwind the agent's call stack when HITL is needed.

    This is NOT an error — it's a control flow mechanism, like a Unix signal.
    The coroutine is destroyed (Python can't pickle it), but the Checkpoint
    survives and contains everything needed to resume.
    """

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint


class ReplayDivergenceError(Exception):
    """Raised when a replayed syscall doesn't match the recorded log.

    This means the agent function is not deterministic — it's making different
    syscall sequences on replay than it did originally. This is always a bug:
    either the agent called an LLM directly (bypassing the proxy) or used
    randomness/time/etc. without going through a syscall.
    """

    def __init__(self, index: int, expected: dict, actual: dict):
        super().__init__(
            f"Replay divergence at syscall #{index}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )


class BudgetExhaustedError(Exception):
    """Raised when an agent tries to use a tool it can't afford."""

    def __init__(self, resource: str, cost: float, remaining: float):
        super().__init__(
            f"Budget exhausted: {resource} — need {cost}, have {remaining}"
        )


# ============================================================================
# PART 2: TOOL REGISTRY
#
# Tools are the "system calls" of our OS. Each tool declares:
#   - Which resource it consumes (like "io", "network", "api_usd")
#   - How much it costs per call
#   - Whether it's destructive (triggers automatic human review)
# ============================================================================


@dataclass
class ToolMeta:
    """Metadata for a registered tool."""

    name: str
    func: Callable  # The actual Python function
    resource: str  # Budget resource type (e.g., "io", "network")
    cost: float = 1.0  # Budget units consumed per call
    destructive: bool = False  # If True, always suspend for human review


# Module-level registry. The @tool decorator writes here.
_registry: dict[str, ToolMeta] = {}


def tool(resource: str, cost: float = 1.0, destructive: bool = False):
    """Decorator to register a function as a mini-castor tool.

    Usage:
        @tool("io")
        async def read_file(path: str) -> str:
            return open(path).read()

        @tool("io", destructive=True)
        async def delete_file(path: str) -> str:
            os.remove(path)
            return f"Deleted {path}"
    """

    def decorator(func: Callable) -> Callable:
        meta = ToolMeta(
            name=func.__name__,
            func=func,
            resource=resource,
            cost=cost,
            destructive=destructive,
        )
        _registry[meta.name] = meta
        return func

    return decorator


# ============================================================================
# PART 3: SYSCALL PROXY
#
# The ONLY interface between an agent function and the kernel.
# Every side effect — every tool call, every LLM inference — must go through
# `await proxy.syscall(tool_name, arguments)`.
#
# The proxy implements three paths:
#
#   REPLAY PATH:  We're re-running the agent after a suspend/resume.
#                 Return the cached response from syscall_log instantly.
#
#   FAST PATH:    New syscall, tool is safe, budget sufficient.
#                 Validate → deduct budget → execute → log → return.
#
#   SLOW PATH:    New syscall, tool is destructive.
#                 Set pending_hitl → raise SuspendInterrupt → agent unwinds.
#                 Human reviews. On resume, we replay from the top.
# ============================================================================


class SyscallProxy:
    """The replay gateway. Injected into every agent function.

    This is the most important class in the entire system.
    """

    def __init__(self, checkpoint: Checkpoint, registry: dict[str, ToolMeta]):
        self.checkpoint = checkpoint
        self._registry = registry
        self._replay_index = 0  # Where we are in the syscall log

    @property
    def is_replaying(self) -> bool:
        """Are we still serving cached responses from a previous run?"""
        return self._replay_index < len(self.checkpoint.syscall_log)

    async def syscall(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """The single entry point for all agent actions.

        This 6-step pipeline is the heart of the kernel:
        """
        request = {"tool_name": tool_name, "arguments": arguments}

        # ── Step 1: REPLAY PATH ──────────────────────────────────────────
        # If we're replaying, serve the cached response. The agent function
        # doesn't know it's being replayed — it gets the same return value
        # as the original execution.
        if self._replay_index < len(self.checkpoint.syscall_log):
            record = self.checkpoint.syscall_log[self._replay_index]
            if record.request != request:
                raise ReplayDivergenceError(
                    self._replay_index, record.request, request
                )
            self._replay_index += 1
            return record.response

        # ── Past this point, we're in LIVE execution (not replay) ────────

        # ── Step 2: TOOL LOOKUP ──────────────────────────────────────────
        if tool_name not in self._registry:
            raise RuntimeError(f"Unknown tool: {tool_name!r}")
        meta = self._registry[tool_name]

        # ── Step 3: SLOW PATH — destructive tools suspend for review ─────
        # This is the HITL gate. Destructive tools ALWAYS suspend, no
        # matter how much budget is available. The human must approve.
        if meta.destructive:
            self.checkpoint.pending_hitl = request
            self.checkpoint.status = "SUSPENDED"
            raise SuspendInterrupt(self.checkpoint)

        # ── Step 4: BUDGET CHECK — deduct before execute ─────────────────
        # We deduct BEFORE executing, so if execution fails, we refund.
        # This prevents a subtle bug: if we deducted after execution and
        # the tool raised an exception, the deduction would stick but the
        # result would never be logged. On replay, the syscall would
        # re-execute and deduct again — a permanent budget leak.
        remaining = self.checkpoint.budgets.get(meta.resource, 0)
        if remaining < meta.cost:
            raise BudgetExhaustedError(meta.resource, meta.cost, remaining)
        self.checkpoint.budgets[meta.resource] = remaining - meta.cost

        # ── Step 5: EXECUTE — run the actual tool ────────────────────────
        try:
            if asyncio.iscoroutinefunction(meta.func):
                result = await meta.func(**arguments)
            else:
                result = meta.func(**arguments)
        except BaseException:
            # Refund on failure — the syscall never completed, so it won't
            # be in the log. Without this refund, replay would re-attempt
            # and re-deduct, causing a permanent leak.
            self.checkpoint.budgets[meta.resource] += meta.cost
            raise

        # ── Step 6: LOG AND RETURN ───────────────────────────────────────
        # Append to the log and advance the replay cursor. On future
        # replays, this response will be served from cache in Step 1.
        self.checkpoint.syscall_log.append(
            SyscallRecord(request=request, response=result)
        )
        self._replay_index = len(self.checkpoint.syscall_log)
        return result


# ============================================================================
# PART 4: HITL HANDLER
#
# When a destructive tool suspends the agent, the human has three choices:
#
#   APPROVE:  Execute the tool as-is. Log with was_hitl=True.
#   REJECT:   Block the action. Log a rejection response so the LLM sees
#             the feedback on replay and can choose a different approach.
#   MODIFY:   The key insight. The human provides natural language feedback
#             (e.g., "only delete files older than 7 days"). We log the
#             ORIGINAL request with the feedback. On replay, the LLM sees
#             the feedback and issues a REVISED syscall. We never mutate
#             the pending request — that would break replay determinism.
# ============================================================================


class HITLHandler:
    """Processes human feedback on suspended agents."""

    async def approve(self, checkpoint: Checkpoint, registry: dict[str, ToolMeta]):
        """Execute the blocked tool and resume."""
        request = checkpoint.pending_hitl
        meta = registry[request["tool_name"]]

        # Deduct and execute (same as proxy fast path)
        checkpoint.budgets[meta.resource] -= meta.cost
        if asyncio.iscoroutinefunction(meta.func):
            result = await meta.func(**request["arguments"])
        else:
            result = meta.func(**request["arguments"])

        checkpoint.syscall_log.append(
            SyscallRecord(request=request, response=result, was_hitl=True)
        )
        checkpoint.pending_hitl = None
        checkpoint.status = "RUNNING"

    def reject(self, checkpoint: Checkpoint, feedback: str):
        """Block the action with human feedback."""
        checkpoint.syscall_log.append(
            SyscallRecord(
                request=checkpoint.pending_hitl,
                response={"status": "REJECTED", "feedback": feedback},
                was_hitl=True,
            )
        )
        checkpoint.pending_hitl = None
        checkpoint.status = "RUNNING"

    def modify(self, checkpoint: Checkpoint, feedback: str):
        """Log the original request with modification feedback.

        WHY NOT just edit the pending request?
        Because on replay, the agent function will emit the ORIGINAL request
        (it doesn't know about the modification). If the log contains a
        MODIFIED request, the replay assertion fails — the requests don't match.

        Instead, we log the original request with HITL_MODIFIED status and the
        human's feedback. On replay, the agent sees this feedback and the LLM
        re-plans with revised arguments. The revised call is a NEW syscall entry.
        """
        checkpoint.syscall_log.append(
            SyscallRecord(
                request=checkpoint.pending_hitl,
                response={"status": "MODIFIED", "feedback": feedback},
                was_hitl=True,
            )
        )
        checkpoint.pending_hitl = None
        checkpoint.status = "RUNNING"


# ============================================================================
# PART 5: CONTEXTVAR BRIDGE
#
# The proxy is powerful, but it creates a coupling problem: every agent
# function must accept a `proxy` parameter. This means agent code "knows"
# about the kernel.
#
# Real Castor solves this with a ContextVar — a thread-local-like variable
# scoped to the current asyncio Task. The kernel sets it before running the
# agent; free functions like call_tool() read it implicitly.
#
# This enables two roles:
#   OPERATOR         — sets up the kernel, registers tools, manages budgets
#   AGENT DEVELOPER  — writes pure logic using call_tool(), budget(), etc.
#                      No kernel imports. No proxy parameter. Zero coupling.
#
# The ContextVar is set in Kernel.run() (Part 6), right before invoking
# the agent function. It's invisible to the agent — just like how libc
# hides raw syscall numbers behind printf() and malloc().
# ============================================================================


# The ContextVar that holds the current proxy. One per asyncio Task.
_current_proxy: ContextVar[SyscallProxy] = ContextVar("mini_castor_proxy")


def _get_proxy() -> SyscallProxy:
    """Internal: retrieve the proxy from ContextVar.

    Raises RuntimeError if called outside Kernel.run().
    """
    try:
        return _current_proxy.get()
    except LookupError:
        raise RuntimeError(
            "call_tool/budget must be called inside Kernel.run()"
        ) from None


async def call_tool(name: str, /, **kwargs: Any) -> Any:
    """Call a registered tool by name — no proxy needed.

    This is the "agent developer" API. The proxy is looked up from a
    ContextVar that the kernel sets automatically.

    Usage:
        async def my_agent():
            result = await call_tool("search", query="hello")
            return result

    In real Castor, this is `castor.lib.tool()`.
    """
    return await _get_proxy().syscall(name, kwargs)


def budget(resource: str) -> float:
    """Check remaining budget for a resource — no proxy needed.

    Usage:
        remaining = budget("api")
        if remaining < 5.0:
            return "Running low on budget"

    In real Castor, this is `castor.lib.budget()`.
    """
    return _get_proxy().checkpoint.budgets.get(resource, 0)


# ============================================================================
# PART 6: THE KERNEL
#
# Ties everything together. Creates checkpoints, runs agent functions,
# handles the three exit modes (completion, suspension, preemption).
#
# The kernel is the "operator" interface — it manages the lifecycle while
# agents run in "user space" oblivious to the machinery underneath.
#
# Key addition in v0.4: the kernel now has approve/reject/modify methods
# directly (instead of exposing hitl as a sub-object), and auto-detects
# the agent's signature to support both classic and new-style agents.
# ============================================================================


class Kernel:
    """The mini-castor kernel. Manages agent lifecycle.

    Usage:
        kernel = Kernel(budgets={"io": 10})
        kernel.register_agent("my_agent", my_agent)
        cp = await kernel.run("my_agent")

        # HITL: approve/reject/modify directly on the kernel
        await kernel.approve(cp)
        kernel.reject(cp, "too risky")
        kernel.modify(cp, "only delete old files")
    """

    def __init__(self, budgets: dict[str, float] | None = None):
        self._budgets = budgets or {}
        self._agents: dict[str, Callable] = {}
        self._registry = dict(_registry)  # Snapshot of registered tools
        self._hitl = HITLHandler()

    def register_agent(self, name: str, fn: Callable):
        """Register an agent function by name."""
        self._agents[name] = fn

    def register_tool(self, meta: ToolMeta):
        """Register a tool directly (alternative to @tool decorator)."""
        self._registry[meta.name] = meta

    async def run(self, agent_name: str, checkpoint: Checkpoint | None = None) -> Checkpoint:
        """Run (or resume) an agent function.

        If checkpoint is None, creates a fresh one.
        If checkpoint is provided (e.g., after HITL), replays from existing log.

        The three exit modes:
          1. COMPLETED — agent function returned normally
          2. SUSPENDED — agent hit a destructive tool (SuspendInterrupt)
          3. PREEMPTED — external cancel via asyncio (CancelledError)
        """
        if agent_name not in self._agents:
            raise RuntimeError(f"Unknown agent: {agent_name!r}")

        if checkpoint is None:
            checkpoint = Checkpoint(
                pid=f"{agent_name}-0",
                budgets=dict(self._budgets),
            )

        agent_fn = self._agents[agent_name]
        proxy = SyscallProxy(checkpoint, self._registry)
        checkpoint.status = "RUNNING"

        # ── Set the ContextVar so call_tool()/budget() work ──────────────
        # This is the bridge between Part 5 (ContextVar) and Part 3 (Proxy).
        # After this line, any call to call_tool() inside the agent will
        # find this proxy via _current_proxy.get().
        _current_proxy.set(proxy)

        # ── Detect agent signature ───────────────────────────────────────
        # 0 required params = new-style agent (uses call_tool/budget)
        # 1+ required params = classic agent (receives proxy explicitly)
        #
        # This lets both styles coexist. In real Castor, the same detection
        # happens in AgentRunner.run().
        sig = inspect.signature(agent_fn)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]

        try:
            if len(required) == 0:
                checkpoint.result = await agent_fn()
            else:
                checkpoint.result = await agent_fn(proxy)
            checkpoint.status = "COMPLETED"
        except SuspendInterrupt:
            pass  # Checkpoint already set by proxy (status=SUSPENDED)
        except asyncio.CancelledError:
            checkpoint.status = "PREEMPTED"
            raise  # Propagate so the caller knows

        return checkpoint

    # ── HITL facade methods ──────────────────────────────────────────────
    # In the old API, callers did: kernel.hitl.approve(cp, kernel._registry)
    # Now they do: kernel.approve(cp) — cleaner, no internal leaking.

    async def approve(self, checkpoint: Checkpoint) -> None:
        """Approve a pending HITL syscall and execute the blocked tool."""
        await self._hitl.approve(checkpoint, self._registry)

    def reject(self, checkpoint: Checkpoint, reason: str) -> None:
        """Reject a pending HITL syscall with feedback."""
        self._hitl.reject(checkpoint, reason)

    def modify(self, checkpoint: Checkpoint, feedback: str) -> None:
        """Approve with modification — log feedback for LLM re-planning."""
        self._hitl.modify(checkpoint, feedback)

    async def run_as_task(self, agent_name: str, checkpoint: Checkpoint | None = None):
        """Run agent as a background asyncio.Task (for preemption demos)."""
        task = asyncio.create_task(self.run(agent_name, checkpoint))
        return task
