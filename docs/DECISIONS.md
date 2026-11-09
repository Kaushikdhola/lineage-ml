# Decisions

Choices made while running slash commands without stopping (see "Autonomy" in CLAUDE.md). Each entry: date,
command, decision, alternatives, how to revert. Newest first.

## 2026-11-09 · /story-polish · three-line copy limit and phone scrolly

**Decision.** "Shorten anything over three lines on a phone" is applied as a hard limit (measured at 390 and
360 px) to text read beside a film: captions, scrolly step paragraphs, the lead and the objective note.
Caveats stay, compressed (for example "(untrained probe)"); details that no longer fit (DeepSpeed PR
numbers, the maintainer's replies on #50419) remain in the panel notes in `data.json` and in
`xray/README.md`. Longer explanatory prose, the `use_cache` box and the colophon were tightened, not cut to
three lines, because cutting them would drop the method's limits. On one-column layouts a scrolly step
activates in a band at 72–92% of the viewport, below the sticky film, instead of 40–60%, where the film
covered the text. Deviations under 10⁻³ print in scientific notation (they read "0.000" before).
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
