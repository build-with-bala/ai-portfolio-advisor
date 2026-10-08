# Vibe Coding Log

**Project:** AI Portfolio Advisor · **Platform:** Claude Code (Anthropic CLI, Claude Opus), one orchestrating session plus one background sub-agent · **Date of this session:** 8 October 2026

This page records the session that brought the repository up to the capstone specification. Commits before this date (the original notebook, the Streamlit app, the live-quote adapters) predate this log.

## Platforms and tools

| Tool | Used for |
|---|---|
| Claude Code (terminal agent) | Reading the spec and the codebase, writing `src/data.py`, `src/model.py`, the Model report tab, tests, the report and deck generator |
| Claude Code sub-agent (background) | Designing and building `landing/index.html` in parallel, with its own brief |
| Playwright (browser automation) | Screenshots to check the landing page and app at desktop and phone widths |
| `uv`, `pytest`, GitHub CLI | Environment, 22 automated tests, repository operations |

## Workflow

1. **Audit before building.** Prompt: *"give me crux"* on the assignment PDF, then *"do i have any projects satisfying all requirements"*. The agent searched local folders and GitHub and scored each project against the rubric row by row. Outcome: this project had the app and SHAP but lacked tuning, cross-validation, the required file layout and this log.
2. **Gap-driven build.** Prompt: *"complete all the requirements for this also send an agent to create a good landing page"*. The agent turned each missing rubric row into a concrete change (table below).
3. **Parallel delegation.** The landing page went to a sub-agent with a written brief: allowed folder, facts it may state, a rule that every performance number stays empty until real metrics exist, a design bar, and a screenshot check before reporting back.
4. **Verify, then report.** Every change was run: the training script end to end, the test suite, and the app rendered through Streamlit's test harness against the new model bundle.

| Rubric gap | Change made by the agent |
|---|---|
| No hyperparameter tuning | Random search, 25 settings, scored by pinball loss |
| No cross-validation | Purged walk-forward CV, 5 expanding folds, 31-day gap |
| Scaling not shown | Ridge baseline with `StandardScaler` fitted per fold; note on why trees need none |
| Look-ahead leakage from fundamentals | Removed fundamentals from model inputs; kept them for the app's separate score |
| Training lived in a notebook | `src/data.py` and `src/model.py`; one command retrains from a committed data snapshot |
| `streamlit_app.py` | Renamed to `app.py`; Dockerfiles and CI updated |

## Prompt strategies that worked

- **Give the rubric, not a task list.** Handing the agent the grading table let it rank work by points (40 for rigor, 20 for defense).
- **Ask for evidence.** "Satisfying all requirements" produced a met / missing table with file and line evidence instead of a yes/no.
- **Brief sub-agents like a colleague.** Scope, facts, forbidden claims and how to verify. The landing page agent was told to leave metrics null instead of inventing them.
- **Demand honest numbers.** The agent reported that the tuned model only matches a naive baseline in cross-validation and that the backtest does not beat the benchmark; those results are in the report unchanged.

## What went wrong, and the fix

- **Ambiguous reference.** "Complete all the requirements for *this*" was read as a different project, and about an hour went into the wrong repository. Lesson: name the repository in the prompt.
- **A silent data bug.** Joining forecasts back onto a date index that repeats once per stock multiplied the rows. A length-mismatch error exposed it; forecasts are now attached by position and a comment in `with_forecasts` records why.
- **Notebook settings were over-complex.** The original 31-leaf, 400-tree model scored 9% worse than the tuned 4-leaf, 100-tree model. Cross-validation, not intuition, settled it.

## What the team owns

AI wrote the scaffolding. The team must be able to explain: pinball loss and why three quantile models, why the folds are purged, why fundamentals were removed, what the mutual-information selection does, what SHAP values mean for one stock, how Hierarchical Risk Parity sizes positions, and why the backtest result means the product is sold on risk ranges, not on returns.
