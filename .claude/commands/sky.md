---
description: Build the home experience, the 3D sky of formulas with semantic zoom into the lab
---

Build the home experience described in CLAUDE.md "Design direction": the sky, semantic zoom and
the continuous camera move into a formula. Use the **frontend-design** and **threejs-** skills.

- Data comes from `src/atlas` (formulas, families, edges, hub flag). Layout: families as clusters,
  positions from a deterministic force layout seeded for reproducibility, hub at the centre with
  filaments into each family.
- Interactions: hover a star → name + family + one-line idea; click → camera flies in, the card
  appears, the formula lifts into the existing lab view (`FormulaStage`) without a page reload. Back
  flies out. Search (⌘K) jumps anywhere. Deep links `?f=<id>` open directly in the lab.
- Idle: slow drift; a "tour" mode that walks a chain (dot product → multi-head) for recording.
- Performance: instanced stars, one draw call for filaments, ≤ 2 ms per frame on a laptop.
- Mobile: a 2D constellation fallback with the same navigation.

Write down 3 directions, pick one by docs/PLAN.md and docs/DESIGN.md, log the choice and the alternatives
in docs/DECISIONS.md, then build, then run `/qa`.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
