---
description: Build a production 3D scene for a formula (or a chain) bound to the symbol bus
argument-hint: <formula id or chain>
---

Build the production scene for: $ARGUMENTS

Read CLAUDE.md "Design direction". Use the **frontend-design** skill for direction and the
**threejs-** skills for implementation (fundamentals, materials, shaders, postprocessing,
interaction, performance).

1. Before coding, propose 3 visual metaphors that make the math itself visible (not decoration).
   For each: what every token in the formula becomes in the scene. Pick the strongest and say why.
2. Implement in `src/scene/` as a class with `update(...)`, `dispose()`, lazy-loaded via dynamic
   import. Every number comes from `src/core`. Bind every token in the formula to a scene object
   through `SymbolBus`, both directions (hover object → symbol lights; hover symbol → object lights).
3. When the user moves to a child formula, the scene must transform continuously, not reset.
4. Reduced-motion and low-power fallback; dispose all GPU resources on navigation; 60 fps target.
5. Register it as the probe for the formula(s) in `src/ui/probes.ts`. Run `npm run ci`.
   Then run `/qa` on it.
