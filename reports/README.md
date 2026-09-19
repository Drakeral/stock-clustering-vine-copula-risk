# Report-ready artifacts

This directory contains small, version-controlled tables, figures and a concise
results summary generated from the passed audit JSON files. They are presentation
artifacts, not an additional modelling stage: the generator performs no model
refits and no new statistical inference.

Regenerate them from the repository root with:

```zsh
uv run python scripts/build_report_artifacts.py
```

`report_manifest.json` binds every generated artifact to its exact audit inputs,
configuration and SHA-256 digest. All outputs deliberately carry the scope
`provisional_research_results` because licensed point-in-time S&P/GICS
reconciliation remains outstanding under protocol amendment PA-001.

The SVG files are the canonical figures. They remain sharp when inserted into a
report and are byte-reproducible under the locked project environment.
