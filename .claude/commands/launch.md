---
description: Final checks, deploy, publish the package, and draft the launch posts
---

1. `npm run ci` and `node scripts/xray.mjs test` must pass. `token-xray report` must match `public/xray/data.json`
   (re-export if the installed versions changed, and say so).
2. Check docs/DEPLOY.md steps 1–6 against the repository and log any gap in docs/DECISIONS.md. Pushing and
   deploying are hard stops: prepare everything, then print the exact commands for me to run
   (`git push origin main`, then watch the Deploy workflow). Once the site is live, a later run of this
   command opens the live story at `/` and checks headers with `curl -sI`, no console errors, films render,
   mobile works.
3. **Package.** Build `xray/` with `python -m build` and run `twine check --strict`. Publishing is a hard stop
   and uses Trusted Publishing (docs/DEPLOY.md, "Publishing the package"), never a token: print
   `git tag xray-v<version> && git push origin xray-v<version>` for me to run. After the Publish workflow
   succeeds, confirm `pip install token-xray-ml` works in a fresh venv and the README example runs.
   Only after that: switch the install command on the story page (`index.html`, `.install`), in the root
   `README.md` (quick start) and in `xray/README.md` from `pip install "git+https://github.com/kaushikdhola/token-xray#subdirectory=xray"`
   to `pip install token-xray-ml`. Until then, keep the git install.
4. **Posts.** Posting is a hard stop: write the drafts to `docs/launch/` and I post them.
   - Reddit r/MachineLearning: title starts with `[P]`, plain and specific, no hype; body explains the
     method in 4 sentences, links the story, the repo and the package, and clearly separates real runs,
     emulations and the known issue (huggingface/transformers#50419). Check the subreddit's current rules first.
   - X thread: 5 posts, the hero film image first, one claim per post, link last.
   - LinkedIn: hook first, under 1,300 characters, a 30–40 second screen recording of the hero flicker,
     the accumulation scrolly and the GRPO morph; link in the first comment.
     Every number in a post must come from `public/xray/data.json` or the tests.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
