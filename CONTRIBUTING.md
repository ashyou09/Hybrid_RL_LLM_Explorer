# How We Work

## The loop for every step

1. **Discuss** — agree on the step's design (in chat, or an issue with the *Level step* template if a teammate should see it).
2. **Build** — small commits.
3. **Check** — the step's "done when" condition from `future_plan/future.md` passes, and `pytest` passes.
4. **Push** — straight to `main`. No pull request is needed: the repo owner pushes directly.
5. **Track** — update `future_plan/tracker.md` (status + commit).

A level is finished only when its write-ups in `conclusion/level_N/` are done and the
handover checklist in `future.md` is ticked.

## Before every push

- [ ] Tests pass (`pytest`)
- [ ] `future_plan/tracker.md` updated
- [ ] No API keys or secrets in the commit

Note: pushing to `main` also updates the Hugging Face Space automatically.

## Commits

Short imperative summary, with a prefix:

| Prefix | For |
|---|---|
| `feat:` | new code |
| `fix:` | bug fix |
| `docs:` | plans, write-ups, README |
| `exp:` | experiment configs and results |
| `test:` | tests only |

## Research rules

1. **Every number has a command.** Results come from scripts in this repo with fixed
   seeds; the command sits next to the number in the write-up.
2. **Hypotheses are not results.** Expected numbers stay labelled *H#* until measured.
3. **Report failures too.** A hypothesis that turns out false is a result.
4. **No secrets in the repo.** API keys come from environment variables.
5. **Only checked references** (✅ in `docs/references.md`) go into a paper.
6. **The v0 demo keeps working.** New research code lives in `research/`.
