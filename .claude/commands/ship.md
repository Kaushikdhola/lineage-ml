---
description: Release checks, deploy to Cloudflare Pages, and prepare launch assets
---

1. `npm run ci` must pass. Fix failures (3 attempts each); anything still failing goes under "Blocked" in
   docs/DECISIONS.md and is not shipped.
2. Check: no placeholder text, no console errors in `npm run preview`, Open Graph and Twitter meta
   tags with an absolute `og:image` (1200×630 rendered from the sky or a morph frame), favicon,
   page titles per formula (`?f=` deep links).
3. Deploying is a hard stop: build, then print `npm run deploy` for me to run (on its first run wrangler
   opens a browser to log in to a free Cloudflare account). After I have deployed, a later run of this
   command sets the live URL as the og:image base, rebuilds, and prints the redeploy command.
4. Record-ready: a `?tour=1` mode that plays a chain morph-by-morph with no UI chrome, for a 40-second
   screen recording at 1080×1350.
5. Draft 3 LinkedIn posts (hook first, under 1,300 characters) and a first comment with the live link
   and the repo, in `docs/launch/`. Claims must match what the site does. Posting is a hard stop.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
