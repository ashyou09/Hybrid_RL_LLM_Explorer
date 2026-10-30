# How We Work

## The loop for every step

1. **Discuss** — open an issue with the *Level step* template; agree on the design.
2. **Build** — on a branch, in small commits.
3. **Push** — open a pull request using the PR template.
4. **Review** — check the "done when" condition from `future_plan/future.md`.
5. **Track** — update `future_plan/tracker.md` (status + commit).

A level is finished only when its write-ups in `conclusion/level_N/` are done and the
handover checklist in `future.md` is ticked.

## Branches

- `main` — stable; it syncs automatically to the Hugging Face Space.
- Work branches — one per step, e.g. `level1/step-1-foundations`.

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
