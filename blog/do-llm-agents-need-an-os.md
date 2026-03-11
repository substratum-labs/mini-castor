# Do LLM Agents Need an OS?

*A 500-line thought experiment in Python*

---

## Intelligence Without Control

LLM agents are getting smarter every month. But the way we run them hasn't changed: a prompt, a while loop, and a prayer.

```python
while not done:
    action = llm.generate(messages, tools=tools)
    result = execute_tool(action.tool_name, action.arguments)  # no isolation
    messages.append(result)
```

This architecture has three gaps:

**No interception point.** If the agent decides to call `delete_database()`, the damage is done before you see it in the logs. There is no gate between the LLM's decision and the side effect.

**No budget.** The agent can make unlimited API calls, send unlimited emails, or consume unlimited compute. The only limit is when the context window fills up or the while loop hits max iterations.

**No crash recovery.** If the process dies, all state is lost. The agent starts from scratch — re-executing every tool call, re-spending every API dollar.

We solved analogous problems decades ago. In the 1960s, computers ran one program at a time with full hardware access. Then operating systems arrived — process isolation, resource quotas, preemptive scheduling — and everything changed.

What if we applied the same ideas to LLM agents?

## The OS Analogy

| OS Concept | Agent Equivalent |
|---|---|
| System calls | All tool calls go through a validated gateway |
| Process checkpoints | Agent state is a replay log, not a serialized coroutine |
| Resource quotas | Finite, depletable budgets per resource type |
| Hardware interrupts | Destructive ops auto-suspend for human review |

The key insight: **the agent function is "user space," and the kernel controls all side effects.**

## Why a Microkernel, Not a Framework

If agents need OS-like controls, the obvious move is to bake them into the agent framework itself. LangChain adds guardrails. CrewAI adds role-based access. Every framework reinvents its own safety layer.

This is the monolithic kernel approach — and it has the same problems it had in the 1980s:

**Tight coupling.** Safety logic is entangled with orchestration logic. You can't use LangChain's budget system with AutoGen's agents. Each framework is a walled garden.

**All or nothing.** Want checkpoint/replay? You need to adopt the entire framework. Want HITL approval? Same deal. There's no way to add just the control layer.

**Growing attack surface.** Every new framework feature is another place where an unchecked tool call can slip through. The more the framework does, the harder it is to audit.

The microkernel approach inverts this. The kernel does exactly four things: validate tool calls, enforce budgets, gate destructive operations, and manage checkpoints. Everything else — orchestration, prompting, LLM selection, agent logic — stays in user space. The kernel is small enough to audit, framework-agnostic, and composable.

```
Monolithic:   [Agent logic + tools + safety + budgets + HITL + replay]
              ← one framework, tightly coupled

Microkernel:  [Agent logic + tools]  ←  user space (any framework)
              ────────────────────────────────────────────────────
              [validation | budgets | HITL | replay]  ←  kernel
```

This is why Mini-Castor exists as a standalone kernel, not a plugin for an existing framework. Any framework can integrate with it. Any agent can run on top of it.

To test this idea, I wrote [Mini-Castor](https://github.com/substratum-labs/mini-castor) — the entire microkernel in one Python file. No dependencies. No frameworks. Just `asyncio`, `dataclasses`, and `contextvars`. ~500 lines.

## The Syscall Proxy

Every agent action goes through a single gateway:

```python
result = await proxy.syscall("search_emails", {"query": "older than 30 days"})
```

The proxy implements three paths:

- **Replay path** — serving cached responses from a previous run (instant)
- **Fast path** — budget check → execute → log (normal execution)
- **Slow path** — destructive tool → suspend for human review

The agent never calls tools directly. It doesn't know which path it's on.

## The Core Trick: Non-Serialized Coroutines

This is the most important design choice.

Python coroutines can't be serialized. You can't pickle an `async def` that's halfway through — it holds C-level stack frames, event loop references, and closure state.

Mini-Castor's solution: **don't serialize the coroutine at all.**

1. Record every syscall and its response in a log
2. To "suspend": raise an exception that unwinds the entire call stack
3. To "resume": re-run the function from the top, serve cached responses
4. The agent fast-forwards through cached syscalls, then continues live

```
Resume after suspension:
  syscall #0: search_emails  → cached (instant)
  syscall #1: analyze        → cached (instant)
  syscall #2: delete_emails  → LIVE (human approved)
  syscall #3: send_summary   → LIVE
```

The agent doesn't know it was ever "killed." From its perspective, `syscall()` just returned a value. This also gives you crash recovery for free — save the checkpoint to disk, resume from any point.

## Capability Budgets

Every tool declares its cost:

```python
@tool("io", cost=2, destructive=True)
async def delete_emails(criteria: str) -> str: ...
```

The kernel deducts before execution and refunds on failure. When the budget hits zero, the agent is stopped. No runaway API bills.

A subtle detail: deduction happens *before* execution. If we deducted after and the tool raised an exception, the cost would stick but the result would never be logged. On replay, the syscall would re-execute and deduct again — a permanent budget leak.

## Why Modify Doesn't Mutate the Request

Destructive tools auto-suspend. The human gets three choices: approve, reject, or modify.

"Modify" is the subtlest design decision. When a human says "only delete files older than 90 days," we do NOT edit the pending request. That would break replay:

```
On replay, the agent emits:  delete(scope="all")       ← original
But the log would contain:   delete(scope="90+ days")  ← mutated
                              MISMATCH → ReplayDivergenceError
```

Instead, we log the original request with the human's feedback as the response. On replay, the agent sees `{"status": "MODIFIED", "feedback": "only 90+ days"}` and the LLM re-plans with revised arguments. The revised call becomes a new syscall entry. Full audit trail. Replay integrity preserved. The human writes natural language, not JSON.

## The ContextVar Bridge

The proxy is powerful, but it couples agent code to the kernel. Every agent must accept a `proxy` parameter. We can do better:

```python
# Agent code — zero kernel imports, zero kernel coupling
async def my_agent():
    result = await call_tool("search", query="hello")
    remaining = budget("api")
    return result
```

A `ContextVar` (Python's async-aware thread-local) holds the current proxy. The kernel sets it before running the agent; free functions read it implicitly. This creates a clean separation:

- **Operator**: sets up kernel, registers tools, manages budgets
- **Agent developer**: writes pure logic using `call_tool()` / `budget()`

Like how libc hides raw syscall numbers behind `printf()` and `malloc()`. The kernel auto-detects the agent's signature — `agent(proxy)` gets the proxy explicitly, `agent()` uses ContextVar implicitly. Both work.

## Try It

```bash
git clone https://github.com/substratum-labs/mini-castor.git
cd mini-castor
python demo.py
```

The demo runs a research assistant that searches, analyzes, and tries to delete emails. The kernel suspends at the destructive call. You choose: approve, reject, or modify. Watch the replay.

No API keys. No dependencies. 30 seconds.

## What This Leaves Out

Mini-Castor is a teaching tool, not a production system. It deliberately skips:

- **Schema validation** — real tools need Pydantic validation with LLM-readable errors
- **Sub-agent spawning** — parent agents delegating budget to children
- **Context window management** — evicting old messages when the window fills
- **Persistence** — saving checkpoints to SQLite for real crash recovery
- **Streaming + preemption** — canceling an LLM mid-generation and capturing partial output

These are hard problems. We're working on a production kernel that tackles them.

## The Question

The agent ecosystem is building increasingly complex orchestration on top of the "prompt + while loop." But maybe what we need isn't another framework on top — it's a layer underneath. A kernel that any framework can integrate with. A syscall boundary that makes every tool call auditable, budgeted, and interruptible.

Do LLM agents need an OS? [Read the code](https://github.com/substratum-labs/mini-castor/blob/main/mini_castor.py) and decide for yourself.

---

*Mini-Castor is open source under Apache 2.0. The entire kernel is [one file](https://github.com/substratum-labs/mini-castor/blob/main/mini_castor.py), ~500 lines, heavily commented. Designed to be read top to bottom in one sitting.*
