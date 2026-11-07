---
description: Final checks, deploy, publish the package, and draft the launch posts
---

1. `npm run ci` and `node scripts/xray.mjs test` must pass. `token-xray report` must match `public/xray/data.json`
   (re-export if the installed versions changed, and say so).
2. Confirm docs/DEPLOY.md steps 1–6 are done (ask me). Push to `main`; watch the Deploy workflow; open the
   live `/story/` and check headers with `curl -sI`, no console errors, films render, mobile works.
3. **Package.** Build `xray/` with `python -m build`; check the README renders; publish to TestPyPI first,
   then PyPI with a token I provide via an environment variable (never write it to disk or the repo).
   Confirm `pip install token-xray` works in a fresh venv and the README example runs.
4. **Posts** (drafts only, I post them):
   - Reddit r/MachineLearning: title starts with `[P]`, plain and specific, no hype; body explains the
     method in 4 sentences, links the story, the repo and the package, and clearly separates real runs,
     emulations and the candidate finding. Check the subreddit's current rules first.
   - X thread: 5 posts, the hero film image first, one claim per post, link last.
   - LinkedIn: hook first, under 1,300 characters, a 30–40 second screen recording of the hero flicker,
     the accumulation scrolly and the GRPO morph; link in the first comment.
     Every number in a post must come from `public/xray/data.json` or the tests.
