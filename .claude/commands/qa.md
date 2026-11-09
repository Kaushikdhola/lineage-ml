---
description: Screenshot, critique like a senior designer and a skeptical researcher, fix, repeat
argument-hint: [optional page or formula ids]
---

Use the **webapp-testing** skill (Python Playwright via `.venv/bin/python`). Scope: $ARGUMENTS
(default: home, softmax, softmax-temperature, scaled-dot-product-attention, multi-head-attention,
ppo-clip, dpo, pass-hat-k).

1. Screenshot each at 1440×900, 1080×1350 and 390×844, plus one frame mid-morph per transition.
2. Collect console errors and page errors. Measure frame time during scene animation.
3. Critique every screenshot twice: as a senior visual designer (hierarchy, typography, spacing,
   clipping, legibility at phone size, motion) and as an ML researcher (is the math correct and
   honestly labelled, does the scene show what the formula says).
4. Fix the issues and repeat. Max 3 rounds. End with a table: issue → fix → verified.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
