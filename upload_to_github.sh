#!/bin/bash
# ===== ArabAgentHallu: رفع المستودع إلى GitHub =====
# قبل التشغيل: أنشئي مستودعًا فارغًا على github.com باسم ArabAgentHallu
# (زر New repository — بدون README ولا .gitignore، فارغ تمامًا)

set -e

# 1) هيكلة المجلد المحلي (شغّليه من المجلد الذي فيه ملفاتك)
mkdir -p generator harness results/tables paper
mv core.py domains.py generate.py build.py generator/ 2>/dev/null || true
mv evaluate.py evaluate_local.py evaluate_batched.py harness/ 2>/dev/null || true
mv raw_cot300.jsonl results/ 2>/dev/null || true
mv table*.csv results/tables/ 2>/dev/null || true
mv ArabAgentHallu_paper_v0_7.pdf paper/ 2>/dev/null || true

# 2) تهيئة git والدفع
git init
git add .
git commit -m "ArabAgentHallu v0.7: suite, generator, harness, measured cot300 results, paper"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ArabAgentHallu.git
git push -u origin main

# 3) وسم إصدار (يلزم لـ Zenodo)
git tag -a v0.7.0 -m "First measured results release"
git push origin v0.7.0
echo "تم الرفع. الآن أنشئي Release من التاغ v0.7.0 على واجهة GitHub."
