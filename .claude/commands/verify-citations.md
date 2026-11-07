---
description: Check every unverified citation against its source and mark it verified or fix it
---

For each formula whose citation has `status: "unverified"` (see `npm run atlas:check`):

1. Open the source (arXiv abstract or PDF, publisher page, or book). Confirm the authors, year, title
   and venue, and that the locator (equation number or section) actually contains this formula in the
   form we show. Note any notation differences.
2. If everything checks out, set `status: "verified"`. If the locator is wrong, correct it and verify.
   If the TeX differs from the source in substance (not just notation), stop and show me both versions.
3. If a source cannot be reached, leave it unverified and list it.

Finish with a table: formula → result (verified / corrected / unreachable) → evidence (the exact
equation or section you found). Run `npm run ci` before committing.
