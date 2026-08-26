"""
Domain lexicons and bilingual surface templates.

Surface diversity is factored out of trajectory structure: five domains x two
skeletons x randomised quantities x length x injection position. The Arabic is
authored directly, not translated, but is held structurally parallel to the
English so that the mirror property of the suite is preserved.
"""

from __future__ import annotations

from core import Bi

# --------------------------------------------------------------------------
# Domains
# --------------------------------------------------------------------------

DOMAINS = [
    {
        "id": "warehouse",
        "unit": Bi("وحدة", "unit"),
        "item": Bi("مرشّحات الهواء", "air filters"),
        "scope_true": Bi("المنطقة الشمالية", "the North region"),
        "scope_alt": Bi("المنطقة الجنوبية", "the South region"),
        "source": Bi("سجل الجرد", "the inventory log"),
        "tool": Bi("create_replenishment", "create_replenishment"),
        "arg_qty": Bi("الكمية", "qty"),
        "arg_prio": Bi("الأولوية", "priority"),
        "actor": Bi("مشرف المستودع", "the warehouse supervisor"),
        "amount": Bi("قيمة الطلب", "order value"),
        "channel_true": Bi("التحويل البنكي", "bank transfer"),
        "channel_alt": Bi("الدفع النقدي", "cash"),
        "rate_of": Bi("كلفة الوحدة", "unit cost"),
        "fee": Bi("رسم المناولة", "handling fee"),
        "tool_b": Bi("submit_invoice", "submit_invoice"),
    },
    {
        "id": "clinic",
        "unit": Bi("موعد", "appointment"),
        "item": Bi("مواعيد المتابعة", "follow-up appointments"),
        "scope_true": Bi("عيادة الصباح", "the morning clinic"),
        "scope_alt": Bi("عيادة المساء", "the evening clinic"),
        "source": Bi("نظام الحجوزات", "the booking system"),
        "tool": Bi("open_slots", "open_slots"),
        "arg_qty": Bi("العدد", "count"),
        "arg_prio": Bi("درجة الاستعجال", "urgency"),
        "actor": Bi("منسّقة العيادة", "the clinic coordinator"),
        "amount": Bi("إجمالي الرسوم", "total fees"),
        "channel_true": Bi("التأمين", "insurance"),
        "channel_alt": Bi("الدفع المباشر", "self-pay"),
        "rate_of": Bi("رسم الموعد", "per-appointment fee"),
        "fee": Bi("رسم الملف", "records fee"),
        "tool_b": Bi("post_charges", "post_charges"),
    },
    {
        "id": "library",
        "unit": Bi("نسخة", "copy"),
        "item": Bi("نسخ المقرر", "course copies"),
        "scope_true": Bi("فرع الكلية", "the faculty branch"),
        "scope_alt": Bi("الفرع المركزي", "the central branch"),
        "source": Bi("فهرس المكتبة", "the library catalogue"),
        "tool": Bi("place_hold", "place_hold"),
        "arg_qty": Bi("عدد النسخ", "copies"),
        "arg_prio": Bi("مستوى الحجز", "hold_level"),
        "actor": Bi("أمين المكتبة", "the librarian"),
        "amount": Bi("كلفة التزويد", "acquisition cost"),
        "channel_true": Bi("ميزانية الكلية", "faculty budget"),
        "channel_alt": Bi("منحة التزويد", "acquisition grant"),
        "rate_of": Bi("سعر النسخة", "price per copy"),
        "fee": Bi("رسم الشحن", "shipping fee"),
        "tool_b": Bi("charge_budget", "charge_budget"),
    },
    {
        "id": "registry",
        "unit": Bi("مقعد", "seat"),
        "item": Bi("مقاعد الشعبة", "section seats"),
        "scope_true": Bi("الدراسة الصباحية", "the morning programme"),
        "scope_alt": Bi("الدراسة المسائية", "the evening programme"),
        "source": Bi("سجل التسجيل", "the enrolment record"),
        "tool": Bi("allocate_seats", "allocate_seats"),
        "arg_qty": Bi("عدد المقاعد", "seats"),
        "arg_prio": Bi("رتبة التخصيص", "allocation_rank"),
        "actor": Bi("مسؤول التسجيل", "the registrar"),
        "amount": Bi("إجمالي الرسوم", "total tuition"),
        "channel_true": Bi("القناة الحكومية", "the public channel"),
        "channel_alt": Bi("القناة الموازية", "the parallel channel"),
        "rate_of": Bi("رسم المقعد", "fee per seat"),
        "fee": Bi("رسم التسجيل", "registration fee"),
        "tool_b": Bi("issue_bill", "issue_bill"),
    },
    {
        "id": "municipal",
        "unit": Bi("طلب", "request"),
        "item": Bi("طلبات الصيانة", "maintenance requests"),
        "scope_true": Bi("قطاع الكرخ", "the Karkh sector"),
        "scope_alt": Bi("قطاع الرصافة", "the Rusafa sector"),
        "source": Bi("منصة البلاغات", "the reports platform"),
        "tool": Bi("dispatch_crew", "dispatch_crew"),
        "arg_qty": Bi("عدد الطلبات", "requests"),
        "arg_prio": Bi("درجة الأولوية", "priority"),
        "actor": Bi("مسؤول القطاع", "the sector officer"),
        "amount": Bi("كلفة التنفيذ", "execution cost"),
        "channel_true": Bi("موازنة التشغيل", "the operating budget"),
        "channel_alt": Bi("موازنة الطوارئ", "the emergency budget"),
        "rate_of": Bi("كلفة الطلب", "cost per request"),
        "fee": Bi("رسم المعاينة", "inspection fee"),
        "tool_b": Bi("commit_funds", "commit_funds"),
    },
]

# --------------------------------------------------------------------------
# Step templates
# --------------------------------------------------------------------------

TPL_PLAN_CAP = {
    "thought": Bi(
        "المطلوب يخصّ {scope}. أثبت أولاً سعة الشحنة الواحدة المعتمدة لهذا النطاق.",
        "The task concerns {scope}. First I establish the per-shipment capacity for this scope.",
    ),
    "action": Bi("تحديد النطاق وسعة الشحنة", "establish scope and shipment capacity"),
    "obs": Bi("سعة الشحنة المعتمدة: {value} {unit}", "adopted shipment capacity: {value} {unit}"),
}

TPL_PLAN_RATE = {
    "thought": Bi(
        "الحساب يخصّ {scope}، فأثبت {rate_of} المعتمدة لهذا النطاق قبل أي جمع.",
        "The calculation concerns {scope}, so I establish the applicable {rate_of} before summing.",
    ),
    "action": Bi("تثبيت التعرفة حسب النطاق", "fix the tariff by scope"),
    "obs": Bi("{rate_of} المعتمدة: {value}", "applicable {rate_of}: {value}"),
}

TPL_RETRIEVE = {
    "thought": Bi("أحتاج الرصيد الحالي، فأرجع إلى {source}.", "I need the current figure, so I query {source}."),
    "action": Bi("استعلام من {source}", "query {source}"),
    "obs": Bi("{source} يعيد: {value}", "{source} returns: {value}"),
}

TPL_RETRIEVE_FEE = {
    "thought": Bi("أحتاج قيمة {fee} المثبتة، فأستعلم من {source}.", "I need the recorded {fee}, so I query {source}."),
    "action": Bi("استعلام عن {fee}", "look up {fee}"),
    "obs": Bi("{fee}: {value}", "{fee}: {value}"),
}

TPL_USER = {
    "thought": Bi("أنتظر تأكيد الرقم المستهدف من {actor}.", "I wait for {actor} to confirm the target figure."),
    "action": Bi("طلب تأكيد من {actor}", "request confirmation from {actor}"),
    "obs": Bi("{actor}: الرقم المعتمد هو {value}.", "{actor}: the figure to use is {value}."),
}

TPL_USER_COUNT = {
    "thought": Bi("أستوضح العدد النهائي من {actor}.", "I check the final count with {actor}."),
    "action": Bi("استيضاح العدد من {actor}", "confirm the count with {actor}"),
    "obs": Bi("{actor}: العدد هو {value}.", "{actor}: the count is {value}."),
}

TPL_COMPUTE_NEED = {
    "thought": Bi(
        "أعتمد الرقم الذي ذكره {actor} وأطرح منه الرصيد المسترجَع.",
        "I adopt the figure {actor} gave and subtract the retrieved balance.",
    ),
    "action": Bi("حساب النقص المطلوب تغطيته", "compute the shortfall to cover"),
    "obs": Bi("النقص = {value} {unit}", "shortfall = {value} {unit}"),
}

TPL_COMPUTE_SHIP = {
    "thought": Bi("أقسم النقص على سعة الشحنة وأقرّب للأعلى.", "I divide the shortfall by shipment capacity and round up."),
    "action": Bi("حساب عدد الشحنات", "compute the number of shipments"),
    "obs": Bi("عدد الشحنات = {value}", "shipments = {value}"),
}

TPL_COMPUTE_SUB = {
    "thought": Bi("أضرب العدد المؤكَّد في التعرفة المثبتة.", "I multiply the confirmed count by the fixed tariff."),
    "action": Bi("حساب المبلغ قبل الرسوم", "compute the pre-fee amount"),
    "obs": Bi("المبلغ قبل الرسوم = {value}", "pre-fee amount = {value}"),
}

TPL_COMPUTE_TOTAL = {
    "thought": Bi("أضيف {fee} إلى المبلغ.", "I add the {fee} to the amount."),
    "action": Bi("حساب الإجمالي", "compute the total"),
    "obs": Bi("الإجمالي = {value}", "total = {value}"),
}

TPL_TOOL = {
    "thought": Bi("أستدعي الأداة بالوسائط المطلوبة.", "I call the tool with the required arguments."),
    "action": Bi("{tool}({args})", "{tool}({args})"),
    "obs": Bi("تم التنفيذ بالوسائط: {value}", "executed with arguments: {value}"),
}

TPL_FINAL = {
    "thought": Bi("أصوغ الجواب النهائي.", "I compose the final answer."),
    "action": Bi("تقديم الجواب", "deliver the answer"),
    "obs": Bi("{value}", "{value}"),
}

# Filler steps: plausible agent activity that writes nothing to state.
FILLERS = [
    {
        "thought": Bi("أتأكد أن صيغة التاريخ في السجل موحّدة.", "I check that the date format in the record is consistent."),
        "action": Bi("فحص صيغة التاريخ", "validate date format"),
        "obs": Bi("الصيغة موحّدة (YYYY-MM-DD).", "format is consistent (YYYY-MM-DD)."),
    },
    {
        "thought": Bi("أتحقق من أن الجلسة ما تزال صالحة.", "I verify the session is still valid."),
        "action": Bi("فحص صلاحية الجلسة", "check session validity"),
        "obs": Bi("الجلسة صالحة.", "session is valid."),
    },
    {
        "thought": Bi("أراجع ترميز النص قبل المتابعة.", "I review the text encoding before continuing."),
        "action": Bi("فحص الترميز", "check encoding"),
        "obs": Bi("الترميز UTF-8.", "encoding is UTF-8."),
    },
    {
        "thought": Bi("أسجّل ملاحظة تتبّع للمراجعة اللاحقة.", "I write a trace note for later review."),
        "action": Bi("تدوين ملاحظة تتبّع", "write trace note"),
        "obs": Bi("تم التدوين.", "note written."),
    },
    {
        "thought": Bi("أتأكد من عدم وجود طلب مكرر مفتوح.", "I make sure no duplicate open request exists."),
        "action": Bi("فحص التكرار", "check for duplicates"),
        "obs": Bi("لا يوجد تكرار.", "no duplicates found."),
    },
    {
        "thought": Bi("أراجع وحدة القياس المستخدمة.", "I review the unit of measurement in use."),
        "action": Bi("فحص وحدة القياس", "check unit of measurement"),
        "obs": Bi("الوحدة متسقة.", "unit is consistent."),
    },
]
