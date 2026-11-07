---
description: Release checks, deploy to Cloudflare Pages, and prepare launch assets
---

1. `npm run ci` must pass. Stop on any failure.
2. Check: no placeholder text, no console errors in `npm run preview`, Open Graph and Twitter meta
   tags with an absolute `og:image` (1200×630 rendered from the sky or a morph frame), favicon,
   page titles per formula (`?f=` deep links).
3. `npm run deploy`. On first run wrangler opens a browser to log in to a free Cloudflare account;
   tell me and wait. Print the live URL, set it as the og:image base, rebuild and redeploy.
4. Record-ready: a `?tour=1` mode that plays a chain morph-by-morph with no UI chrome, for a 40-second
   screen recording at 1080×1350.
5. Draft 3 LinkedIn posts (hook first, under 1,300 characters) and a first comment with the live link
   and the repo. Claims must match what the site does.
