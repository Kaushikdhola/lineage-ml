---
description: Prepare a maintainer-ready report for the packed position_ids / use_cache behaviour
---

1. Run `xray/repro/packed_position_ids_use_cache.py` with the latest released transformers and with the
   version in `public/xray/data.json`. Record both results.
2. Search the transformers GitHub issues and pull requests for existing reports (terms: packed,
   padding-free, position_ids, use_cache, DynamicCache, find_packed_sequence_indices). Search the docs for
   any statement that `use_cache=False` is required for padding-free training. Record what you find in the
   final report and in docs/DECISIONS.md.
3. If it is not already reported or documented, write `docs/report-transformers-use-cache.md`: a short
   title, environment, the minimal reproduction, expected vs observed, the code path
   (`masking_utils._preprocess_mask_arguments` only detects packed sequences when `past_key_values is
None`; models create a `DynamicCache` when `use_cache` is true), why the Trainer is unaffected, and a
   suggested fix (skip cache creation in training mode, or detect packing regardless, or warn).
4. Filing or commenting upstream is a hard stop: never open the issue yourself. Prepare the text, then print
   the exact text and the URL for me to file it. Keep the tone factual and appreciative; no claims about
   impact beyond what the reproduction shows.
5. If it IS already known or documented, update the story's finding box and the README to say so and link
   it, and drop the word "candidate".

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
