"""
ArabAgentHallu — core execution engine.

Design principle
----------------
A trajectory is not authored as text. It is authored as a small *program*: an
ordered list of operations over a shared state dict. Rendering to Arabic /
English natural language happens only at the end, from the executed values.

Three properties follow for free:

1. Propagation is automatic. Injecting a hallucination = overriding the output
   of one op, then re-executing the remainder. Every downstream step that reads
   the corrupted slot is corrupted consistently, with no manual rewriting.

2. Ground truth is derived, not asserted. The responsible step is *verified*
   against the counterfactual definition (Sec. 3 of the paper) by actually
   running the counterfactuals, not by trusting the injection bookkeeping.

3. The Arabic and English mirrors are guaranteed structurally identical,
   because they are two renderings of one executed program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

Lang = str  # "ar" | "en"


# --------------------------------------------------------------------------
# Bilingual strings
# --------------------------------------------------------------------------

class Bi:
    """A string that exists in both languages. The unit of mirrored authoring."""

    __slots__ = ("ar", "en")

    def __init__(self, ar: str, en: str) -> None:
        self.ar = ar
        self.en = en

    def s(self, lang: Lang) -> str:
        return self.ar if lang == "ar" else self.en

    def fmt(self, lang: Lang, **kw: Any) -> str:
        return self.s(lang).format(**kw)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Bi({self.ar!r}, {self.en!r})"


def txt(v: Any, lang: Lang) -> str:
    """Render a value for display in a given language."""
    if isinstance(v, Bi):
        return v.s(lang)
    if isinstance(v, bool):
        return ("نعم" if v else "لا") if lang == "ar" else ("yes" if v else "no")
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, dict):
        sep = "، " if lang == "ar" else ", "
        return sep.join(f"{k}={txt(x, lang)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        sep = "، " if lang == "ar" else ", "
        return sep.join(txt(x, lang) for x in v)
    return str(v)


# --------------------------------------------------------------------------
# State references
# --------------------------------------------------------------------------

class Ref:
    """A read from the trajectory state, resolved at execution time."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"Ref({self.name!r})"


def resolve(v: Any, state: Dict[str, Any]) -> Any:
    if isinstance(v, Ref):
        if v.name not in state:
            raise KeyError(f"unresolved reference {v.name!r}")
        return state[v.name]
    if isinstance(v, dict):
        return {k: resolve(x, state) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve(x, state) for x in v]
    return v


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

KINDS = ("plan", "retrieve", "user", "compute", "tool", "verify", "finalize")

# Which injection category may target which op kind.
CATEGORY_OF_KIND = {
    "plan": "planning",
    "retrieve": "retrieval",
    "compute": "reasoning",
    "user": "human-interaction",
    "tool": "tool-use",
}


@dataclass
class Op:
    """One agent step, as an executable operation."""

    kind: str
    out: Optional[str] = None                       # state slot written
    params: Dict[str, Any] = field(default_factory=dict)
    fn: Optional[Callable[[Dict[str, Any]], Any]] = None   # for compute/finalize
    tpl: Dict[str, Bi] = field(default_factory=dict)       # thought/action/obs
    filler: bool = False                            # does not affect the answer
    injectable: bool = False                        # a valid injection target

    @property
    def category(self) -> Optional[str]:
        return CATEGORY_OF_KIND.get(self.kind)


def exec_op(op: Op, state: Dict[str, Any]) -> Any:
    """Execute one op against the current state, returning its output value."""
    p = resolve(op.params, state)
    if op.kind in ("plan", "retrieve", "user", "verify"):
        return p.get("value")
    if op.kind == "tool":
        return dict(p.get("args", {}))
    if op.kind in ("compute", "finalize"):
        if op.fn is None:
            raise ValueError(f"{op.kind} op requires fn")
        return op.fn(p)
    raise ValueError(f"unknown op kind {op.kind!r}")


def run(
    ops: List[Op],
    overrides: Optional[Dict[int, Any]] = None,
    forced: Optional[Dict[int, Any]] = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Execute a program.

    overrides : injected corruptions {step_index: corrupted_value}
    forced    : counterfactual corrections, applied *after* overrides, so a
                forced value at step t restores that step regardless of injection
    """
    state: Dict[str, Any] = {}
    outs: List[Any] = []
    for i, op in enumerate(ops):
        v = exec_op(op, state)
        if overrides and i in overrides:
            v = overrides[i]
        if forced and i in forced:
            v = forced[i]
        outs.append(v)
        if op.out:
            state[op.out] = v
    return state, outs


# --------------------------------------------------------------------------
# Counterfactual attribution ground truth
# --------------------------------------------------------------------------

def responsible_steps(
    ops: List[Op],
    overrides: Dict[int, Any],
    clean_outs: List[Any],
    gold_answer: Any,
) -> Set[int]:
    """
    H(tau) = { t : correcting step t flips the outcome to correct }.

    This is the paper's counterfactual definition, computed by actually running
    each counterfactual rather than assuming the injection site is the answer.
    """
    h: Set[int] = set()
    # The final answer-emitting step is excluded: overwriting the answer itself
    # trivially "fixes" any trajectory, so it is a degenerate member of H that
    # carries no diagnostic information.
    for t in range(len(ops) - 1):
        state, _ = run(ops, overrides=overrides, forced={t: clean_outs[t]})
        if state.get("answer") == gold_answer:
            h.add(t)
    return h


class BuildError(RuntimeError):
    """Raised when a generated item fails its own validity checks."""


def make_item(
    ops: List[Op],
    query: Bi,
    domain: str,
    skeleton: str,
    category: Optional[str] = None,
    target_index: Optional[int] = None,
    corrupt_value: Any = None,
    item_id: str = "",
) -> Dict[str, Any]:
    """
    Build one validated record (clean if category is None, else hallucinated).

    Every hallucinated item is checked against three conditions before it is
    allowed out of the generator:
      (a) the injection actually changes the final answer,
      (b) the injection site is a member of H(tau),
      (c) the injection site is the *earliest* member, i.e. t* = min H(tau).
    An item failing any check is rejected, not silently emitted.
    """
    clean_state, clean_outs = run(ops)
    gold = clean_state.get("answer")
    if gold is None:
        raise BuildError("program produced no answer slot")

    if category is None:
        return _record(
            ops, query, domain, skeleton, clean_outs, gold,
            is_hallucination=False, t_star=None, category=None,
            gold_answer=gold, item_id=item_id,
        )

    if target_index is None:
        raise BuildError("hallucinated item requires target_index")

    overrides = {target_index: corrupt_value}
    bad_state, bad_outs = run(ops, overrides=overrides)
    produced = bad_state.get("answer")

    if produced == gold:
        raise BuildError("injection did not change the final answer (a)")

    h = responsible_steps(ops, overrides, clean_outs, gold)
    if target_index not in h:
        raise BuildError("injection site is not hallucination-responsible (b)")
    if min(h) != target_index:
        raise BuildError(
            f"injection site {target_index} is not earliest in H={sorted(h)} (c)"
        )

    return _record(
        ops, query, domain, skeleton, bad_outs, produced,
        is_hallucination=True, t_star=target_index, category=category,
        gold_answer=gold, item_id=item_id, h_set=sorted(h),
        clean_outs=clean_outs,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_steps(ops: List[Op], outs: List[Any], lang: Lang) -> List[Dict[str, str]]:
    """Render executed ops as thought / action / observation triplets."""
    state: Dict[str, Any] = {}
    steps: List[Dict[str, str]] = []
    for i, (op, v) in enumerate(zip(ops, outs), start=1):
        kw = {k: txt(resolve(x, state), lang) for k, x in op.params.items()}
        kw["value"] = txt(v, lang)
        if op.kind == "tool":
            # the action must display the *executed* arguments, so that a
            # corrupted argument is visible where the agent commits it
            kw["args"] = kw["value"]
        steps.append({
            "step": i,
            "thought": op.tpl["thought"].fmt(lang, **kw),
            "action": op.tpl["action"].fmt(lang, **kw),
            "observation": op.tpl["obs"].fmt(lang, **kw),
        })
        if op.out:
            state[op.out] = v
    return steps


def _record(
    ops, query, domain, skeleton, outs, produced, *,
    is_hallucination, t_star, category, gold_answer, item_id,
    h_set=None, clean_outs=None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": item_id,
        "domain": domain,
        "skeleton": skeleton,
        "n_steps": len(ops),
        "category": category,
        "is_hallucination": is_hallucination,
        "hallucination_step": (t_star + 1) if t_star is not None else None,
        "responsible_steps": [t + 1 for t in h_set] if h_set else None,
        "step_kinds": [op.kind for op in ops],
        "filler_steps": [i + 1 for i, op in enumerate(ops) if op.filler],
        "produced_answer": {l: txt(produced, l) for l in ("ar", "en")},
        "gold_answer": {l: txt(gold_answer, l) for l in ("ar", "en")},
        "languages": {},
    }
    for lang in ("ar", "en"):
        rec["languages"][lang] = {
            "query": query.s(lang),
            "steps": render_steps(ops, outs, lang),
        }
    if t_star is not None and clean_outs is not None:
        rec["gold_explanation"] = {
            l: _explain(ops[t_star], outs[t_star], clean_outs[t_star], t_star + 1, l)
            for l in ("ar", "en")
        }
    return rec


_EXPLAIN = {
    "planning": Bi(
        "في الخطوة {n} اعتُمد فرض تخطيطي خاطئ ({bad} بدل {good})، فانحرفت كل "
        "الحسابات اللاحقة المبنية عليه.",
        "At step {n} the agent adopted a wrong planning premise ({bad} instead of "
        "{good}); every downstream computation inherited it.",
    ),
    "retrieval": Bi(
        "في الخطوة {n} أعادت عملية الاسترجاع قيمة ({bad}) تناقض المرجع المضمَّن في "
        "الطلب ({good})، واعتُمدت دون تحقق.",
        "At step {n} retrieval returned a value ({bad}) contradicting the reference "
        "embedded in the query ({good}), and it was used unverified.",
    ),
    "reasoning": Bi(
        "في الخطوة {n} وقع خطأ حسابي: النتيجة {bad} بينما الصحيح {good}.",
        "At step {n} the arithmetic is wrong: it produced {bad} where {good} is correct.",
    ),
    "human-interaction": Bi(
        "في الخطوة {n} أدخل المستخدم معلومة خاطئة ({bad} بدل {good})، وتبنّاها الوكيل "
        "لاحقاً دون مقابلتها بالمرجع المضمَّن. الخطوة المسؤولة هي خطوة الدخول لا خطوة التبنّي.",
        "At step {n} the user introduced false information ({bad} instead of {good}) "
        "which the agent later endorsed without checking it against the embedded "
        "reference. The responsible step is where it enters, not where it is endorsed.",
    ),
    "tool-use": Bi(
        "في الخطوة {n} استُدعيت الأداة بوسيط خاطئ ({bad} بدل {good}).",
        "At step {n} the tool was called with a wrong argument ({bad} instead of {good}).",
    ),
}


def _explain(op: Op, bad: Any, good: Any, n: int, lang: Lang) -> str:
    cat = op.category or "reasoning"
    return _EXPLAIN[cat].fmt(lang, n=n, bad=txt(bad, lang), good=txt(good, lang))


# --------------------------------------------------------------------------
# Small helpers used by domain builders
# --------------------------------------------------------------------------

def digit_slip(n: int, rng) -> int:
    """A plausible arithmetic slip: transpose two digits, or shift one by one."""
    s = str(int(n))
    if len(s) >= 2:
        i = rng.randrange(len(s) - 1)
        if s[i] != s[i + 1]:
            out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            if int(out) != int(n):
                return int(out)
    i = rng.randrange(len(s))
    d = int(s[i])
    nd = (d + rng.choice([-1, 1])) % 10
    out = s[:i] + str(nd) + s[i + 1:]
    val = int(out) if out else int(n) + 1
    return val if val != int(n) else int(n) + 1


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(float(a) / float(b)))
