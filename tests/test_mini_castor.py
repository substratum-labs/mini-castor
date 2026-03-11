"""
Tests for mini-castor.

These tests serve double duty:
  1. Verify correctness of the kernel
  2. Demonstrate each concept in isolation (read them as examples!)
"""

from __future__ import annotations

import asyncio

import pytest

from mini_castor import (
    BudgetExhaustedError,
    Checkpoint,
    HITLHandler,
    Kernel,
    ReplayDivergenceError,
    SuspendInterrupt,
    SyscallProxy,
    SyscallRecord,
    ToolMeta,
    budget,
    call_tool,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_registry(*tools):
    """Build a tool registry from ToolMeta instances."""
    return {t.name: t for t in tools}


def safe_tool(name="read", resource="io", cost=1.0):
    """A non-destructive tool that returns a predictable value."""
    async def _fn(**kwargs):
        return f"{name}({kwargs})"
    return ToolMeta(name=name, func=_fn, resource=resource, cost=cost)


def destructive_tool(name="delete", resource="io", cost=1.0):
    """A destructive tool that should trigger HITL suspension."""
    async def _fn(**kwargs):
        return f"{name}({kwargs})"
    return ToolMeta(name=name, func=_fn, resource=resource, cost=cost, destructive=True)


def make_checkpoint(budgets=None):
    return Checkpoint(pid="test-0", budgets=budgets or {"io": 10})


# ============================================================================
# CONCEPT 1: Basic syscall execution (fast path)
# ============================================================================


class TestFastPath:
    """The happy path: safe tool, sufficient budget, no HITL."""

    async def test_simple_syscall(self):
        """A single syscall executes and is logged."""
        registry = make_registry(safe_tool())
        cp = make_checkpoint()
        proxy = SyscallProxy(cp, registry)

        result = await proxy.syscall("read", {"path": "x.txt"})

        assert result == "read({'path': 'x.txt'})"
        assert len(cp.syscall_log) == 1
        assert cp.syscall_log[0].request == {
            "tool_name": "read", "arguments": {"path": "x.txt"}
        }

    async def test_budget_deducted(self):
        """Each syscall deducts from the budget."""
        registry = make_registry(safe_tool(cost=3.0))
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)

        await proxy.syscall("read", {"path": "a"})
        assert cp.budgets["io"] == 7.0

        await proxy.syscall("read", {"path": "b"})
        assert cp.budgets["io"] == 4.0

    async def test_budget_exhausted(self):
        """Exceeding the budget raises BudgetExhaustedError."""
        registry = make_registry(safe_tool(cost=5.0))
        cp = make_checkpoint({"io": 3})
        proxy = SyscallProxy(cp, registry)

        with pytest.raises(BudgetExhaustedError):
            await proxy.syscall("read", {"path": "x"})

        # Nothing logged — the syscall never executed
        assert len(cp.syscall_log) == 0

    async def test_budget_refund_on_failure(self):
        """If a tool raises, the budget is refunded."""
        async def failing_tool(**kwargs):
            raise RuntimeError("tool failed!")

        meta = ToolMeta(name="fail", func=failing_tool, resource="io", cost=2.0)
        registry = make_registry(meta)
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)

        with pytest.raises(RuntimeError, match="tool failed"):
            await proxy.syscall("fail", {})

        # Budget refunded — back to original
        assert cp.budgets["io"] == 10.0
        assert len(cp.syscall_log) == 0

    async def test_unknown_tool(self):
        """Calling an unregistered tool raises RuntimeError."""
        proxy = SyscallProxy(make_checkpoint(), {})

        with pytest.raises(RuntimeError, match="Unknown tool"):
            await proxy.syscall("nonexistent", {})


# ============================================================================
# CONCEPT 2: Checkpoint/Replay
# ============================================================================


class TestReplay:
    """The core insight: replay serves cached responses from the log."""

    async def test_replay_returns_cached_responses(self):
        """On replay, the proxy returns cached results without executing."""
        call_count = 0

        async def counting_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        meta = ToolMeta(name="count", func=counting_tool, resource="io", cost=1)
        registry = make_registry(meta)

        # First run: execute two syscalls
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)
        r1 = await proxy.syscall("count", {"n": 1})
        r2 = await proxy.syscall("count", {"n": 2})
        assert r1 == "result-1"
        assert r2 == "result-2"
        assert call_count == 2

        # Replay: create a new proxy with the SAME checkpoint
        # The tool function should NOT be called again
        proxy2 = SyscallProxy(cp, registry)
        r1_replay = await proxy2.syscall("count", {"n": 1})
        r2_replay = await proxy2.syscall("count", {"n": 2})
        assert r1_replay == "result-1"  # Same result
        assert r2_replay == "result-2"  # Same result
        assert call_count == 2  # Tool was NOT called again!

    async def test_replay_then_live(self):
        """After replay exhausts the log, new syscalls execute live."""
        async def echo(**kwargs):
            return f"live({kwargs})"

        meta = ToolMeta(name="echo", func=echo, resource="io", cost=1)
        registry = make_registry(meta)

        # Pre-populate log with one cached entry
        cp = make_checkpoint({"io": 10})
        cp.syscall_log.append(
            SyscallRecord(
                request={"tool_name": "echo", "arguments": {"x": 1}},
                response="cached(1)",
            )
        )

        proxy = SyscallProxy(cp, registry)

        # First call: served from cache
        r1 = await proxy.syscall("echo", {"x": 1})
        assert r1 == "cached(1)"

        # Second call: live execution
        r2 = await proxy.syscall("echo", {"x": 2})
        assert r2 == "live({'x': 2})"

    async def test_replay_divergence_detected(self):
        """If the agent makes a different syscall on replay, we catch it."""
        cp = make_checkpoint()
        cp.syscall_log.append(
            SyscallRecord(
                request={"tool_name": "read", "arguments": {"path": "a"}},
                response="data-a",
            )
        )

        proxy = SyscallProxy(cp, make_registry(safe_tool()))

        # Agent tries a DIFFERENT syscall than what's in the log
        with pytest.raises(ReplayDivergenceError):
            await proxy.syscall("read", {"path": "DIFFERENT"})

    async def test_is_replaying_property(self):
        """is_replaying is True during replay, False after."""
        cp = make_checkpoint({"io": 10})
        cp.syscall_log.append(
            SyscallRecord(
                request={"tool_name": "read", "arguments": {}},
                response="cached",
            )
        )

        registry = make_registry(safe_tool())
        proxy = SyscallProxy(cp, registry)

        assert proxy.is_replaying is True
        await proxy.syscall("read", {})
        assert proxy.is_replaying is False


# ============================================================================
# CONCEPT 3: HITL (Human-in-the-Loop)
# ============================================================================


class TestHITL:
    """Destructive tools suspend. Humans approve, reject, or modify."""

    async def test_destructive_tool_suspends(self):
        """A destructive tool raises SuspendInterrupt."""
        registry = make_registry(destructive_tool())
        cp = make_checkpoint()
        proxy = SyscallProxy(cp, registry)

        with pytest.raises(SuspendInterrupt) as exc_info:
            await proxy.syscall("delete", {"target": "important.txt"})

        assert cp.status == "SUSPENDED"
        assert cp.pending_hitl is not None
        assert cp.pending_hitl["tool_name"] == "delete"
        assert exc_info.value.checkpoint is cp

    async def test_approve_executes_and_logs(self):
        """Approve: execute the tool and log with was_hitl=True."""
        registry = make_registry(destructive_tool())
        cp = make_checkpoint({"io": 10})
        cp.pending_hitl = {"tool_name": "delete", "arguments": {"target": "x"}}
        cp.status = "SUSPENDED"

        handler = HITLHandler()
        await handler.approve(cp, registry)

        assert cp.status == "RUNNING"
        assert cp.pending_hitl is None
        assert len(cp.syscall_log) == 1
        assert cp.syscall_log[0].was_hitl is True
        assert "delete" in str(cp.syscall_log[0].response)

    async def test_reject_logs_feedback(self):
        """Reject: log a rejection response so the LLM can re-plan."""
        cp = make_checkpoint()
        cp.pending_hitl = {"tool_name": "delete", "arguments": {"target": "x"}}
        cp.status = "SUSPENDED"

        handler = HITLHandler()
        handler.reject(cp, "Too dangerous")

        assert cp.status == "RUNNING"
        assert cp.pending_hitl is None
        assert len(cp.syscall_log) == 1
        assert cp.syscall_log[0].response == {
            "status": "REJECTED",
            "feedback": "Too dangerous",
        }
        assert cp.syscall_log[0].was_hitl is True

    async def test_modify_preserves_original_request(self):
        """Modify: log original request + feedback. Never mutate the request."""
        cp = make_checkpoint()
        cp.pending_hitl = {"tool_name": "delete", "arguments": {"target": "all"}}
        cp.status = "SUSPENDED"

        handler = HITLHandler()
        handler.modify(cp, "Only delete files older than 7 days")

        record = cp.syscall_log[0]
        # The ORIGINAL request is preserved in the log
        assert record.request["arguments"]["target"] == "all"
        # The feedback is in the response
        assert record.response == {
            "status": "MODIFIED",
            "feedback": "Only delete files older than 7 days",
        }


# ============================================================================
# CONCEPT 4: Full round-trip (suspend → HITL → replay → complete)
# ============================================================================


class TestFullRoundTrip:
    """End-to-end: agent runs, suspends, human responds, agent replays."""

    async def test_approve_round_trip(self):
        """Agent suspends → approve → replay → completes."""
        call_log = []

        async def logging_read(**kwargs):
            call_log.append(("read", kwargs))
            return "data"

        async def logging_delete(**kwargs):
            call_log.append(("delete", kwargs))
            return "deleted"

        registry = make_registry(
            ToolMeta(name="read", func=logging_read, resource="io", cost=1),
            ToolMeta(name="delete", func=logging_delete, resource="io", cost=1, destructive=True),
        )

        async def agent(proxy: SyscallProxy):
            r = await proxy.syscall("read", {"path": "x"})
            d = await proxy.syscall("delete", {"path": "x"})
            return f"{r}+{d}"

        # Phase 1: run until suspension
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)
        cp.status = "RUNNING"
        try:
            await agent(proxy)
        except SuspendInterrupt:
            pass

        assert cp.status == "SUSPENDED"
        assert len(call_log) == 1  # Only read was called
        assert call_log[0] == ("read", {"path": "x"})

        # Phase 2: human approves
        handler = HITLHandler()
        await handler.approve(cp, registry)
        assert cp.status == "RUNNING"
        assert len(call_log) == 2  # Delete was executed by approve

        # Phase 3: replay
        call_log.clear()
        proxy2 = SyscallProxy(cp, registry)
        cp.status = "RUNNING"
        result = await agent(proxy2)

        # read and delete were served from cache — NOT called again
        assert call_log == []
        assert result == "data+deleted"
        assert cp.status == "RUNNING"  # Agent function returned normally

    async def test_reject_round_trip(self):
        """Agent suspends → reject → replay → agent sees rejection."""
        async def noop(**kwargs):
            return "ok"

        async def danger(**kwargs):
            return "destroyed"

        registry = make_registry(
            ToolMeta(name="safe", func=noop, resource="io", cost=1),
            ToolMeta(name="danger", func=danger, resource="io", cost=1, destructive=True),
            ToolMeta(name="report", func=noop, resource="io", cost=1),
        )

        async def agent(proxy: SyscallProxy):
            await proxy.syscall("safe", {})
            result = await proxy.syscall("danger", {"target": "db"})
            # Agent checks if it was rejected
            if isinstance(result, dict) and result.get("status") == "REJECTED":
                await proxy.syscall("report", {"msg": "blocked"})
                return "aborted"
            return "completed"

        # Phase 1: suspend
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)
        try:
            await agent(proxy)
        except SuspendInterrupt:
            pass

        # Phase 2: reject
        HITLHandler().reject(cp, "Not safe")

        # Phase 3: replay
        proxy2 = SyscallProxy(cp, registry)
        result = await agent(proxy2)

        assert result == "aborted"
        assert len(cp.syscall_log) == 3  # safe + danger(rejected) + report

    async def test_modify_round_trip(self):
        """Agent suspends → modify → replay → agent re-plans."""
        async def noop(**kwargs):
            return f"ok({kwargs})"

        registry = make_registry(
            ToolMeta(name="search", func=noop, resource="io", cost=1),
            ToolMeta(name="delete", func=noop, resource="io", cost=1, destructive=True),
        )

        async def agent(proxy: SyscallProxy):
            await proxy.syscall("search", {"q": "old emails"})
            result = await proxy.syscall("delete", {"scope": "all"})
            if isinstance(result, dict) and result.get("status") == "MODIFIED":
                # LLM re-plans with feedback
                result = await proxy.syscall("delete", {"scope": "older than 90 days"})
            return result

        # Phase 1: suspend at delete
        cp = make_checkpoint({"io": 10})
        proxy = SyscallProxy(cp, registry)
        try:
            await agent(proxy)
        except SuspendInterrupt:
            pass

        # Phase 2: modify
        HITLHandler().modify(cp, "Only delete older than 90 days")

        # Phase 3: replay — the revised delete goes through HITL again!
        proxy2 = SyscallProxy(cp, registry)
        try:
            result = await agent(proxy2)
        except SuspendInterrupt:
            # The REVISED delete also triggers HITL (it's still destructive)
            # This is correct — the human approved the intent, not the execution
            await HITLHandler().approve(cp, registry)

            # Phase 4: replay again with the approved revised delete
            proxy3 = SyscallProxy(cp, registry)
            result = await agent(proxy3)

        assert "older than 90 days" in str(result)


# ============================================================================
# CONCEPT 5: Kernel integration
# ============================================================================


class TestKernel:
    """The Kernel ties proxy + HITL + agent lifecycle together."""

    async def test_kernel_run_to_completion(self):
        """Agent with no destructive calls completes normally."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())

        async def agent(proxy):
            return await proxy.syscall("read", {"path": "x"})

        kernel.register_agent("agent", agent)
        cp = await kernel.run("agent")

        assert cp.status == "COMPLETED"
        assert cp.result == "read({'path': 'x'})"

    async def test_kernel_suspend_and_resume(self):
        """Kernel handles suspend → HITL → resume cycle."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())
        kernel.register_tool(destructive_tool())

        async def agent(proxy):
            a = await proxy.syscall("read", {"x": 1})
            b = await proxy.syscall("delete", {"x": 2})
            return f"{a}+{b}"

        kernel.register_agent("agent", agent)

        # Run: suspends at delete
        cp = await kernel.run("agent")
        assert cp.status == "SUSPENDED"

        # Approve and resume — using kernel facade methods
        await kernel.approve(cp)
        cp = await kernel.run("agent", cp)
        assert cp.status == "COMPLETED"
        assert "+" in cp.result

    async def test_kernel_preemption(self):
        """Kernel supports preemption via task cancellation."""
        kernel = Kernel(budgets={"io": 100})
        kernel.register_tool(safe_tool())

        async def slow_agent(proxy):
            for i in range(100):
                await proxy.syscall("read", {"i": i})
                await asyncio.sleep(0.01)
            return "done"

        kernel.register_agent("slow", slow_agent)

        task = await kernel.run_as_task("slow")
        await asyncio.sleep(0.05)  # Let it run a few syscalls
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_kernel_reject_facade(self):
        """Kernel.reject() works as a facade over HITLHandler."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())
        kernel.register_tool(destructive_tool())

        async def agent(proxy):
            await proxy.syscall("read", {})
            result = await proxy.syscall("delete", {"x": 1})
            if isinstance(result, dict) and result.get("status") == "REJECTED":
                return "rejected"
            return "completed"

        kernel.register_agent("agent", agent)

        cp = await kernel.run("agent")
        assert cp.status == "SUSPENDED"

        kernel.reject(cp, "too risky")
        cp = await kernel.run("agent", cp)
        assert cp.status == "COMPLETED"
        assert cp.result == "rejected"

    async def test_kernel_modify_facade(self):
        """Kernel.modify() works as a facade over HITLHandler."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())
        kernel.register_tool(destructive_tool())

        async def agent(proxy):
            result = await proxy.syscall("delete", {"scope": "all"})
            if isinstance(result, dict) and result.get("status") == "MODIFIED":
                return f"modified: {result['feedback']}"
            return "completed"

        kernel.register_agent("agent", agent)

        cp = await kernel.run("agent")
        assert cp.status == "SUSPENDED"

        kernel.modify(cp, "only old files")

        cp = await kernel.run("agent", cp)
        assert cp.status == "COMPLETED"
        assert cp.result == "modified: only old files"


# ============================================================================
# CONCEPT 6: ContextVar bridge (new-style agents)
# ============================================================================


class TestContextVar:
    """New-style agents use call_tool()/budget() via ContextVar — no proxy param."""

    async def test_call_tool_works_via_contextvar(self):
        """call_tool() finds the proxy from ContextVar, set by kernel.run()."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())

        async def agent():  # <-- No proxy parameter!
            return await call_tool("read", path="x.txt")

        kernel.register_agent("agent", agent)
        cp = await kernel.run("agent")

        assert cp.status == "COMPLETED"
        assert cp.result == "read({'path': 'x.txt'})"

    async def test_budget_works_via_contextvar(self):
        """budget() reads remaining budget from ContextVar proxy."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool(cost=3.0))

        async def agent():
            await call_tool("read", path="a")
            remaining = budget("io")
            return f"remaining={remaining}"

        kernel.register_agent("agent", agent)
        cp = await kernel.run("agent")

        assert cp.status == "COMPLETED"
        assert cp.result == "remaining=7.0"

    async def test_new_style_hitl_round_trip(self):
        """New-style agent goes through full HITL suspend/resume cycle."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())
        kernel.register_tool(destructive_tool())

        async def agent():
            a = await call_tool("read", x=1)
            b = await call_tool("delete", x=2)
            return f"{a}+{b}"

        kernel.register_agent("agent", agent)

        # Run: suspends at destructive call
        cp = await kernel.run("agent")
        assert cp.status == "SUSPENDED"

        # Approve and resume
        await kernel.approve(cp)
        cp = await kernel.run("agent", cp)
        assert cp.status == "COMPLETED"
        assert "+" in cp.result

    async def test_call_tool_outside_kernel_raises(self):
        """call_tool() outside Kernel.run() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be called inside"):
            await call_tool("anything", x=1)

    async def test_dual_signature_classic_still_works(self):
        """Classic agents (with proxy param) still work after ContextVar changes."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())

        async def classic_agent(proxy):
            return await proxy.syscall("read", {"path": "works"})

        kernel.register_agent("classic", classic_agent)
        cp = await kernel.run("classic")

        assert cp.status == "COMPLETED"
        assert "works" in cp.result

    async def test_classic_agent_can_also_use_call_tool(self):
        """Classic agents can use call_tool() too — ContextVar is always set."""
        kernel = Kernel(budgets={"io": 10})
        kernel.register_tool(safe_tool())

        async def hybrid_agent(proxy):
            # Use proxy directly
            a = await proxy.syscall("read", {"path": "via-proxy"})
            # Also use call_tool — both work!
            b = await call_tool("read", path="via-contextvar")
            return f"{a} | {b}"

        kernel.register_agent("hybrid", hybrid_agent)
        cp = await kernel.run("hybrid")

        assert cp.status == "COMPLETED"
        assert "via-proxy" in cp.result
        assert "via-contextvar" in cp.result
