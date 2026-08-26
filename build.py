"""
Skeletons, injectors, and the sampler.

Two structural skeletons cover all five hallucination categories each, so every
category is exercised at every trajectory length and every injection position.
That is what makes per-cell support scale with n instead of staying at 1.

Length and injection position are *controlled*, not incidental: filler steps are
placed before or after the injection site to move t* into a target position
bucket. Without this, injection sites cluster wherever the skeleton happens to
put them, and a model with a positional prior scores well for the wrong reason.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from core import Bi, BuildError, Op, Ref, ceil_div, digit_slip, make_item
from domains import (
    DOMAINS, FILLERS, TPL_COMPUTE_NEED, TPL_COMPUTE_SHIP, TPL_COMPUTE_SUB,
    TPL_COMPUTE_TOTAL, TPL_FINAL, TPL_PLAN_CAP, TPL_PLAN_RATE, TPL_RETRIEVE,
    TPL_RETRIEVE_FEE, TPL_TOOL, TPL_USER, TPL_USER_COUNT,
)

CATEGORIES = ("planning", "retrieval", "reasoning", "human-interaction", "tool-use")
BUCKETS = ("early", "mid", "late")


# --------------------------------------------------------------------------
# Skeleton A — shortfall / shipments
# --------------------------------------------------------------------------

def skeleton_a(rng: random.Random, dom: Dict[str, Any]):
    cap = rng.choice([40, 50, 60, 75, 80, 120])
    cap_alt = rng.choice([c for c in [30, 45, 90, 100, 150] if c != cap])
    on_hand = rng.randrange(12, 90)
    target = on_hand + rng.randrange(60, 400)
    prio = rng.choice([1, 2, 3])
    prio_alt = rng.choice([p for p in [1, 2, 3, 4, 5] if p != prio])

    need = target - on_hand
    if ceil_div(need, cap) == ceil_div(need, cap_alt):
        raise BuildError("planning injection would be answer-neutral")

    core = [
        Op(kind="plan", out="cap", injectable=True, tpl=TPL_PLAN_CAP,
           params={"value": cap, "scope": dom["scope_true"], "unit": dom["unit"]}),
        Op(kind="retrieve", out="on_hand", injectable=True, tpl=TPL_RETRIEVE,
           params={"value": on_hand, "source": dom["source"]}),
        Op(kind="user", out="target", injectable=True, tpl=TPL_USER,
           params={"value": target, "actor": dom["actor"]}),
        Op(kind="compute", out="need", injectable=True, tpl=TPL_COMPUTE_NEED,
           fn=lambda p: p["target"] - p["on_hand"],
           params={"target": Ref("target"), "on_hand": Ref("on_hand"),
                   "actor": dom["actor"], "unit": dom["unit"]}),
        Op(kind="compute", out="ships", tpl=TPL_COMPUTE_SHIP,
           fn=lambda p: ceil_div(p["need"], p["cap"]),
           params={"need": Ref("need"), "cap": Ref("cap")}),
        Op(kind="tool", out="ticket", injectable=True, tpl=TPL_TOOL,
           params={"tool": dom["tool"], "args": {"qty": Ref("need"), "priority": prio}}),
        Op(kind="finalize", out="answer", tpl={
            "thought": TPL_FINAL["thought"], "action": TPL_FINAL["action"],
            "obs": Bi("النقص {need} {unit}، عدد الشحنات {ships}، والطلب المنشأ: {ticket}",
                      "shortfall {need} {unit}, shipments {ships}, request created: {ticket}")},
           fn=lambda p: (p["need"], p["ships"], p["ticket"]["priority"]),
           params={"need": Ref("need"), "ships": Ref("ships"),
                   "ticket": Ref("ticket"), "unit": dom["unit"]}),
    ]

    query = Bi(
        "لدينا {item} في {scope}. سعة الشحنة الواحدة {cap} {unit} في {scope}، و{cap_alt} {unit} "
        "في {scope_alt}. آخر جرد موثّق سجّل {on_hand} {unit}، والسياسة تثبّت الرصيد المستهدف عند "
        "{target} {unit}، والأولوية المطلوبة {prio}. احسب النقص وعدد الشحنات، وأنشئ الطلب.",
        "We handle {item} in {scope}. Per-shipment capacity is {cap} {unit} in {scope} and "
        "{cap_alt} {unit} in {scope_alt}. The last audited count recorded {on_hand} {unit}, "
        "policy fixes the target balance at {target} {unit}, and the requested priority is "
        "{prio}. Compute the shortfall and the shipment count, and create the request.",
    )
    q = Bi(*[query.s(l).format(
        item=dom["item"].s(l), scope=dom["scope_true"].s(l), scope_alt=dom["scope_alt"].s(l),
        unit=dom["unit"].s(l), cap=cap, cap_alt=cap_alt, on_hand=on_hand, target=target, prio=prio,
    ) for l in ("ar", "en")])

    injections = {
        "planning": (0, cap_alt),
        "retrieval": (1, digit_slip(on_hand, rng)),
        "human-interaction": (2, digit_slip(target, rng)),
        "reasoning": (3, digit_slip(need, rng)),
        "tool-use": (5, {"qty": need, "priority": prio_alt}),
    }
    return core, q, injections


# --------------------------------------------------------------------------
# Skeleton B — tariff / total / channel
# --------------------------------------------------------------------------

def skeleton_b(rng: random.Random, dom: Dict[str, Any]):
    rate = rng.choice([15, 18, 22, 25, 30, 45])
    rate_alt = rng.choice([r for r in [12, 20, 28, 35, 50] if r != rate])
    count = rng.randrange(7, 60)
    fee = rng.randrange(5, 95)

    core = [
        Op(kind="plan", out="rate", injectable=True, tpl=TPL_PLAN_RATE,
           params={"value": rate, "scope": dom["scope_true"], "rate_of": dom["rate_of"]}),
        Op(kind="user", out="count", injectable=True, tpl=TPL_USER_COUNT,
           params={"value": count, "actor": dom["actor"]}),
        Op(kind="retrieve", out="fee", injectable=True, tpl=TPL_RETRIEVE_FEE,
           params={"value": fee, "source": dom["source"], "fee": dom["fee"]}),
        Op(kind="compute", out="sub", injectable=True, tpl=TPL_COMPUTE_SUB,
           fn=lambda p: p["count"] * p["rate"],
           params={"count": Ref("count"), "rate": Ref("rate")}),
        Op(kind="compute", out="total", tpl=TPL_COMPUTE_TOTAL,
           fn=lambda p: p["sub"] + p["fee_v"],
           params={"sub": Ref("sub"), "fee_v": Ref("fee"), "fee": dom["fee"]}),
        Op(kind="tool", out="ticket", injectable=True, tpl=TPL_TOOL,
           params={"tool": dom["tool_b"],
                   "args": {"amount": Ref("total"), "channel": dom["channel_true"]}}),
        Op(kind="finalize", out="answer", tpl={
            "thought": TPL_FINAL["thought"], "action": TPL_FINAL["action"],
            "obs": Bi("الإجمالي {total}، والتسوية عبر: {ticket}",
                      "total {total}, settled through: {ticket}")},
           fn=lambda p: (p["total"], p["ticket"]["channel"]),
           params={"total": Ref("total"), "ticket": Ref("ticket")}),
    ]

    query = Bi(
        "أحتاج احتساب {amount} في {scope}. {rate_of} هي {rate} في {scope} و{rate_alt} في "
        "{scope_alt}. الطلب المثبّت يذكر {count} {unit}، و{fee} المسجّل {fee_v}، والتسوية تتم عبر "
        "{channel}. احسب الإجمالي وسجّل العملية.",
        "I need {amount} for {scope}. The {rate_of} is {rate} in {scope} and {rate_alt} in "
        "{scope_alt}. The filed request states {count} {unit}, the recorded {fee} is {fee_v}, "
        "and settlement goes through {channel}. Compute the total and record the transaction.",
    )
    q = Bi(*[query.s(l).format(
        amount=dom["amount"].s(l), scope=dom["scope_true"].s(l), scope_alt=dom["scope_alt"].s(l),
        rate_of=dom["rate_of"].s(l), rate=rate, rate_alt=rate_alt, count=count,
        unit=dom["unit"].s(l), fee=dom["fee"].s(l), fee_v=fee, channel=dom["channel_true"].s(l),
    ) for l in ("ar", "en")])

    injections = {
        "planning": (0, rate_alt),
        "human-interaction": (1, digit_slip(count, rng)),
        "retrieval": (2, digit_slip(fee, rng)),
        "reasoning": (3, digit_slip(count * rate, rng)),
        "tool-use": (5, {"amount": count * rate + fee, "channel": dom["channel_alt"]}),
    }
    return core, q, injections


SKELETONS = {"A": skeleton_a, "B": skeleton_b}
FINALIZE_SLOT = 6  # index of the finalize op inside every core skeleton


# --------------------------------------------------------------------------
# Filler placement / position control
# --------------------------------------------------------------------------

def _place_fillers(
    core: List[Op], target_idx: Optional[int], n_fillers: int,
    bucket: str, rng: random.Random,
) -> Tuple[List[Op], Optional[int]]:
    """Insert filler steps so that t* lands in the requested position bucket."""
    if target_idx is None:
        before = rng.randrange(0, n_fillers + 1)
    elif bucket == "early":
        before = 0
    elif bucket == "late":
        before = n_fillers
    else:
        before = n_fillers // 2
    after = n_fillers - before

    prepend = [0] * len(core)
    hi = target_idx if target_idx is not None else FINALIZE_SLOT - 1
    for _ in range(before):
        prepend[rng.randrange(0, max(hi, 0) + 1)] += 1
    for _ in range(after):
        lo = (target_idx + 1) if target_idx is not None else 0
        prepend[rng.randrange(min(lo, FINALIZE_SLOT), FINALIZE_SLOT + 1)] += 1

    ops: List[Op] = []
    new_target = None
    for i, op in enumerate(core):
        for _ in range(prepend[i]):
            f = rng.choice(FILLERS)
            ops.append(Op(kind="verify", out=None, filler=True, tpl=f, params={"value": ""}))
        if target_idx is not None and i == target_idx:
            new_target = len(ops)
        ops.append(op)
    return ops, new_target


def bucket_of(idx: int, n: int) -> str:
    r = idx / max(n - 1, 1)
    return "early" if r < 1 / 3 else ("mid" if r < 2 / 3 else "late")


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_item(
    rng: random.Random,
    category: Optional[str],
    bucket: str,
    n_fillers: int,
    item_id: str,
) -> Dict[str, Any]:
    """Sample one validated item, or raise BuildError for the caller to retry."""
    dom = rng.choice(DOMAINS)
    skel_name = rng.choice(list(SKELETONS))
    core, query, injections = SKELETONS[skel_name](rng, dom)

    if category is None:
        ops, _ = _place_fillers(core, None, n_fillers, bucket, rng)
        return make_item(ops, query, dom["id"], skel_name, item_id=item_id)

    core_idx, corrupt = injections[category]
    ops, target = _place_fillers(core, core_idx, n_fillers, bucket, rng)
    rec = make_item(
        ops, query, dom["id"], skel_name, category=category,
        target_index=target, corrupt_value=corrupt, item_id=item_id,
    )
    rec["position_bucket"] = bucket_of(target, len(ops))
    return rec
