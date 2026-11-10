# Decisions

Choices made while running slash commands without stopping (see "Autonomy" in CLAUDE.md). Each entry: date,
command, decision, alternatives, how to revert. Newest first.

## 2026-11-09 · package name · token-xray-ml

**Decision.** PyPI refuses `token-xray` as too similar to the existing `tokenxray`, so the distribution is
`token-xray-ml`. The brand stays Token X-ray, the import stays `token_xray` and the command stays
`token-xray`; only `pip install token-xray-ml` changes.
**Alternatives.** `trainer-xray`, `gradient-xray`, `token-radiograph` (all free). **Revert.** Change
`name` in `xray/pyproject.toml` and the install lines.

## 2026-11-09 · package release · Trusted Publishing

**Decision.** `token-xray` is published by `.github/workflows/publish.yml` on `xray-v*` tags through PyPI
Trusted Publishing (OIDC), with signed attestations, instead of `twine upload` with an API token: nothing
long-lived to leak, and every release is tied to a tagged, tested commit. The package description drops
"through the unmodified trainer" (the runs change a few defaults, as the methods section says) and gains
project URLs.
**Alternatives.** `twine upload` with a project-scoped token. **Revert.** Delete the workflow and follow
the old `/launch` step 3.

## 2026-11-09 · story redesign · reading room

**Decision.** The story moves from a light lightbox page with hand-drawn grease-pencil loops to a dim
reading room where the films are the only light: a film renderer with an exposure ramp, halation, grain and
a vignette; scanner-style corner print; bracketed regions of interest with numeric readouts in one caliper
accent; window and level on weight films (drag, arrow keys, double-click to reset); Newsreader for the
essay and IBM Plex Sans Condensed for everything the instrument prints. The hero becomes title plus a
full-width film exposed by a scanning bar. No number, label or run kind changed. docs/DESIGN.md is
rewritten to match, so later polish passes keep it. Atkinson Hyperlegible and Reenie Beanie are removed
(nothing else used them).
**Alternatives.** Keep the light page and only restyle the films; a light page with a dark hero only.
**Revert.** `git revert` this commit.

## 2026-11-09 · renaming · github.com/kaushikdhola/token-xray and tokenxray.pages.dev

**Decision (the user's, on Claude's recommendation).** The launch is Token X-ray, so the repository and the site
carry that name: the repository was renamed to `token-xray` (GitHub redirects the old URLs), and the site
moves to a new Cloudflare Pages project, `tokenxray`. `token-xray.pages.dev` belongs to an unrelated site.
The PyPI package stays `token-xray-ml`, and the atlas stays at `/atlas/`. Every link in the repository
points to the new names; dated records (this file, `docs/redteam/`) keep the names they had. The old
`lineage-ml` Pages project serves only a 301 redirect to the new site, preserving paths, because the PyPI
pages of 0.1.0 and 0.1.1 link to it permanently. The trusted publisher on PyPI must name the new repository.
**Alternatives.** Keep `lineage-ml` (the URL would not match the announced name); a custom domain (costs
money; can be added to the `tokenxray` project later). **Revert.** Rename the repository back on GitHub and
restore the URLs with `git revert`.

## 2026-11-09 · validating the published package · three install bugs, fixed in 0.1.2

**How it was validated.** `token-xray-ml` 0.1.1 installed from PyPI with no cache into a fresh venv:
checksums match PyPI, each file carries a Trusted Publishing provenance attestation from `publish.yml` in
the `pypi` environment, and the wheel's Python files equal the tagged source. The README example passes,
and a planted mean-of-micro-batch-means loop is caught with the exact closed-form weights. The shipped tests
pass against the installed copy (50 passed; the two 4.45.2 tests skip without the side environment).

**Bugs found (all in 0.1.0 and 0.1.1).**

1. `token-xray report` crashed on every pip install: it built the 4.45.2 panel, whose interpreter it looked
   for at a path that exists only inside this repository. Now `report` leaves that panel out and says how
   to include it; `export` (the site's data) still requires it.
2. `report` also built the TRL panels, which it never prints, so a plain install without `[trl]` would
   fail on `import trl`. It now builds only what it prints.
3. Given a 4.45.2 interpreter, an installed copy put its whole `site-packages` on the side interpreter's
   `PYTHONPATH`, so the "4.45.2" run imported transformers 5.22.2 and measured the fixed Trainer
   (reproduced with 0.1.1). The side interpreter now sees only the `token_xray` package.
4. The package did not declare `accelerate`, which the transformers Trainer requires; installs without
   `[trl]` failed. Added `accelerate>=1.1`.

Each fix has a test, and three new faults in `scripts/mutate.py` (`report-requires-legacy`,
`report-builds-trl`, `side-path-exposes-site-packages`) are each caught. The repository's own `.venv` had the
PyPI copy installed over the editable one, so a bare `pytest` in `xray/` tested the download, not the
source; `node scripts/xray.mjs test` was unaffected (it puts `xray/src` first). The PyPI copy was removed
from `.venv`.
**Alternatives.** Make `report` require the side environment (fails for every user). **Revert.** `git revert`
the commits of this entry.

## 2026-11-09 · /launch · install command and the 0.1.1 release

**What happened.** `xray-v0.1.0` had already been tagged and pushed at `58f2f50`, and the Publish
workflow put `token-xray-ml` 0.1.0 on PyPI (2026-11-10 02:26 UTC). That ran before this command's fix to
`publish.yml`, so the release skipped the 4.45.2 legacy test. No code under `xray/src` changed between the
tag and this command, so the published code is the code tested here. `pip install token-xray-ml` works
in a fresh venv, and the README example passes.

**Decision.** Switch the story's install line and `xray/README.md` to `pip install token-xray-ml`, keeping
the git install as the way to get unreleased code. Prepare 0.1.1 (version bump only), because the 0.1.0
page on PyPI shows the old README: it says the package "is not on PyPI yet" and its example has a `...`
placeholder. PyPI never accepts a version twice, so a new release is the only way to fix that page. The
0.1.1 tag also runs the full release checks, including the legacy test.
**Alternatives.** Leave the 0.1.0 page as it is (launch visitors read a false sentence); yank 0.1.0 (would
break installs that pinned it). **Revert.** Set the version back to 0.1.0 in `xray/pyproject.toml` and
`token_xray/__init__.py`, and restore the git install line.

## 2026-11-09 · /launch · deploy and publish audit

**Gaps found and fixed.**

1. `publish.yml` ran the tests without the transformers 4.45.2 side environment, so `test_legacy.py` (the
   hero plate's check) was skipped, not run, on a release. It now builds `.venv-hf445` exactly as
   `xray.yml` does, and `pytest -rs` prints any skip.
2. `docs/DEPLOY.md` step 5 asked for a required pull request or status check. GitHub applies those rules
   to direct pushes, so they would reject the push-to-`main` workflow the deploy relies on. Step 5 now
   asks only for "Restrict deletions" and "Block force pushes", and describes the pull-request setup as an
   opt-in. The Deploy workflow already runs `npm run ci` before publishing.
3. Step 2's command line alternative now passes `--production-branch main`.

**Not checkable from the repository.** Two-factor authentication, the Cloudflare token's scope, the
repository secrets, the ruleset, Dependabot and secret scanning, and the PyPI pending publisher all live in
account settings. The live site on 2026-11-09 served the current build with the security headers, and
`/story/` returned 301 to `/`.

**Measurements.** The installed versions equal those in `data.json` (torch 2.15.0, transformers 5.22.2,
TRL 1.17.1). A fresh export matched `public/xray/data.json` exactly, so it was not re-exported.

**Package.** `token-xray-ml` 0.1.0 builds, `twine check --strict` passes, and the built wheel installs in
a fresh venv; the README example, now filled in so it runs as written, passes `assert_token_mean()`. Not on
PyPI yet, so the story and README keep the git install.

**Posts.** Drafts are in `docs/launch/`, which is kept local and gitignored (removed from the repository on 2026-11-09 at the user's request). Reddit blocked automated reads of r/MachineLearning's rules, and
third-party guides disagree, so `reddit.md` asks you to check the rules before posting.
**Alternatives.** Skip the legacy test on release (ships an unchecked hero claim). **Revert.** `git revert`
the commits of this entry.

## 2026-11-09 · URL layout · the story becomes the home page

**Decision (chosen by the user).** Token X-ray is served at `/` (source `index.html`); the atlas moved to
`/atlas/` (source `atlas/index.html`). `/story` and `/story/` redirect with 301 (`public/_redirects`). Old atlas
links carry a query (`/?f=dpo`, `?tour=1`, `?sky=2d`, `?ablation=1`), which `_redirects` cannot match, so the
story's script forwards them to `/atlas/` with the query and hash intact. Both pages have a canonical link.
In-page links on the story scroll without writing `#section` into the address bar and move focus to the
target; a pasted `/#objectives` still opens at that section. The sky and the lab both carry a "Token X-ray"
chip in their header as the way back (it replaces the line in the sky's legend).
**Alternatives.** Redirect `/` to `/story/` (smallest change, but the address bar always shows `/story/` and the
atlas has no home URL); keep `#section` in the URL (shareable, but the user wanted clean URLs).
**Revert.** `git revert` the two commits of this entry, and delete the Cloudflare redirect if one was added in
the dashboard.

## 2026-11-09 · /story-polish · three-line copy limit and phone scrolly

**Decision.** "Shorten anything over three lines on a phone" is applied as a hard limit (measured at 390 and
360 px) to text read beside a film: captions, scrolly step paragraphs, the lead and the objective note.
Caveats stay, compressed (for example "(untrained probe)"); details that no longer fit (DeepSpeed PR
numbers, the maintainer's replies on #50419) remain in the panel notes in `data.json` and in
`xray/README.md`. Longer explanatory prose, the `use_cache` box and the colophon were tightened, not cut to
three lines, because cutting them would drop the method's limits. On one-column layouts a scrolly step
activates in a band at 72–92% of the viewport, below the sticky film, instead of 40–60%, where the film
covered the text. One exception: the dropped-micro-batch step runs to four lines on a phone, because cutting "the other three micro-batches go dark" left readers unsure whether the black film was a rendering fault. Deviations under 10⁻³ print in scientific notation (they read "0.000" before).
**Alternatives.** Cut every paragraph to three lines (loses honesty caveats); overlay step cards on the film
(hides the film). **Revert.** `git revert` the copy-pass and mobile commits of this date.

## 2026-11-09 · leftovers after /redteam-xray · bf16 bounds in `test_sft_bf16_default_rounds_but_does_not_reweight`

**Decision.** Derive the bounds from bf16's unit roundoff u = 2⁻⁸ ≈ 3.9·10⁻³ instead of picking numbers:
per token |w·N − 1| ≤ 2u (one bf16 rounding forward, one backward), per sequence |mean(w·N) − 1| ≤ u.

**What actually happened (honest timeline).**

1. The sequence-mean bound was first set to 2·10⁻³ _before_ measuring, as a guess.
2. The first measurement was 2.09·10⁻³, so the test failed.
3. The bound was then raised to 5·10⁻³ _after_ measuring. The reason given in the test comment and in the
   report was wrong: it blamed the shortest sequence (2 targets), but the largest deviation is on a
   6-target sequence.
4. Replaced on 2026-11-09 by the derived bounds above. The target count does not tighten the bound: if
   rounding errors were independent, a 6-target mean would stay within u/√6 = 1.6·10⁻³, but it measured
   2.09·10⁻³, so errors within a sequence are correlated and the worst case (a mean is no larger than one
   token's error) applies.

**Measured (test batch, transformers 5.22.2, TRL 1.17.1).** Largest per-token deviation 3.61·10⁻³ (0.92u,
bound 2u = 7.8·10⁻³); largest sequence-mean deviation 2.09·10⁻³ (0.54u, bound u = 3.9·10⁻³).

**The bound still catches a re-weighting.** A 1% re-weighting moves a sequence mean by 10⁻², 2.6× the bound.
Shown twice: the test rejects a 1%-re-weighted copy of the same measurement, and the library fault
`sft-normaliser-1pct` (SFT's token count 1% too small, `xray/tests/conftest.py`) fails it in
`scripts/mutate.py`.

**Alternatives.** A bound shrinking as u/√n (rejected: the measurement shows correlated errors); keeping
5·10⁻³ (rejected: not derived, and its justification was wrong).

**Revert.** Restore the test from commit `e31113b`
(`git show e31113b:xray/tests/test_trl.py`), and drop `sft-normaliser-1pct` from `conftest.py` and
`scripts/mutate.py`.

## 2026-11-09 · /redteam-xray fixes · repository URL and install command

**Decision.** The instruction gave `<MY-GITHUB-USERNAME>`; filled from the git remote:
`https://github.com/kaushikdhola/lineage-ml`. The repository returns 404 to logged-out visitors today, so
the link and the git install work for readers only once it is public.
**Alternatives.** Leave the placeholder (a broken page). **Revert.** Edit the two links in `story/index.html`
and the install block in `xray/README.md`.

## 2026-11-09 · /redteam-xray fixes · one commit for all fixes

**Decision.** The fixes went into one commit (`e31113b`): the page reads panels (`hf445-ga`, `hf-pad-eos`)
that the code change introduces, so splitting code from copy would leave a broken story at the intermediate
commit. The TRL and data-parallel work and the review itself are separate commits.
**Alternatives.** Split by file group. **Revert.** `git revert e31113b`.

## 2026-11-09 · /redteam-xray fixes · hero plate

**Decision.** Real run of the transformers 4.45.2 Trainer in `.venv-hf445` (first attempt succeeded, within
the 45-minute timebox), replacing the emulation; the emulation panel stays in the data for the tests.
**Alternatives.** Relabel the emulation. **Revert.** Point the hero in `src/story/main.ts` back to
`emu-mean-of-means` and restore the lead and meta text.

## 2026-11-09 · /redteam-xray fixes · mutation test in CI

**Decision.** `scripts/mutate.py` takes about 25 minutes on CPU, so the CI job runs on pushes to `main`,
nightly and on demand, not on pull requests.
**Alternatives.** Every PR (slow), or a sampled subset of faults. **Revert.** Remove the `mutation` job
from `.github/workflows/xray.yml`.

## 2026-11-08 · /xray-ddp · Trainer batch and torchrun rendezvous

**Decision.** The Trainer DDP runs use 6 per device so one optimizer step covers all 12 sequences (with 3,
a step covers only 6). torchrun uses a static rendezvous on 127.0.0.1 because `--standalone` hung on this
machine, whose host name resolves to an unreachable reverse-DNS address.
**Alternatives.** Two optimizer steps per run; `--standalone`. **Revert.** Edit `scenarios.ddp_panel`
and `ddp.hf_trainer_ddp`.

## Blocked

Nothing is blocked.
