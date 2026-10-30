# Results

```
results/<level>/
    runs/<timestamp>_<name>/   every run (git-ignored: regenerate with the command in meta.json)
    figures/                   curated plots and replays used in the write-ups (committed)
    tables/                    curated result tables (committed)
```

Each run folder holds `config.json` (all knobs), `meta.json` (command, git commit,
Python version, time) and `metrics.json`. A number in a write-up must point to the
run that produced it.
