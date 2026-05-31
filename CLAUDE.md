# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run main.py                 # run entry point
uv run python -c "..."         # one-off script in the venv
uv add <package>               # add dependency (updates pyproject.toml + uv.lock)
```

## Rules

- do not make decisions on architecture, design, or workaround without explicitly consulting me
- do not add fallbacks without explicitly consulting me
- do not go beyound the scope of the ask unless specified
