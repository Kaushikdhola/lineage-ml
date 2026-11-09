---
description: Wire the story into the site, make the share image, and polish copy and mobile
---

Read docs/DESIGN.md and use the **frontend-design** skill. Keep the radiograph system exactly; do not add
new colours, fonts or decorative motion.

1. **Entry points.** Link the story from the home page (the sky or the lab) with one quiet line, and link
   back to the atlas from the story's colophon. Fix the colophon's GitHub link to the real repository URL.
2. **Share image.** Render a 1200×630 `public/og/story.png` from the hero film (Playwright via the
   **webapp-testing** skill, at device scale 2, then resize). Add absolute `og:image`, `og:url` and
   `twitter:image` to `story/index.html` using the production URL `https://lineage-ml.pages.dev`.
3. **Copy pass.** Every sentence must be true for the measured versions. Shorten anything over three
   lines on a phone. Keep sentence case.
4. **Mobile pass.** 390×844 and 360×740: film notes readable, sticky film not covering the step text,
   slider usable with a thumb, no horizontal scroll.
5. **Accessibility.** Each film has a meaningful `aria-label` (already generated); add a visually hidden
   table of the per-sequence mean weights for screen readers; check keyboard focus order and visible focus.
6. **Performance.** Lighthouse-style check: the story's JS stays under 40 kB gzipped excluding KaTeX;
   films redraw only on resize or data change.
   Run `npm run ci`, then `/qa` with scope `/story/`.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
