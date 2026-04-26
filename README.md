# LangGraph Recruiter Agent

A portfolio demo showing how to wrap an LLM in a small, explicit graph of
deterministic steps using [LangGraph](https://github.com/langchain-ai/langgraph).
Given a free-text role brief, the pipeline parses it into structured filters,
pulls candidates from a MySQL table of parsed LinkedIn profiles, ranks them
with an LLM, and synthesizes a cited shortlist for a hiring manager.

![Streamlit demo screenshot](docs/streamlit_demo.png)

## Why this is interesting
- **LangGraph is the plumbing, not the magic.** Each node has a narrow job so
  the system is testable and easy to reason about.
- **LLM + SQL together.** Structured filters from the LLM are passed into
  parameterized SQL, so ranking only runs over rows that already pass
  hard constraints.
- **Graceful degradation.** Every node has a deterministic fallback: no
  `OPENAI_API_KEY`, no JSON output, no DB match — the graph still returns
  something meaningful and surfaces the reason in the UI.
- **Shippable demo surface.** Streamlit front end with parsed filters,
  ranked candidate cards, and a shortlist summary that cites `profile_id`s.

## Architecture

```mermaid
flowchart LR
    userQuery["User query (Streamlit)"] --> parseQuery[parse_query]
    parseQuery -->|"structured filters"| retrieveCandidates[retrieve_candidates]
    retrieveCandidates -->|"top-K rows (MySQL)"| rankCandidates[rank_candidates]
    rankCandidates -->|"scored + rationales"| synthesizeShortlist[synthesize_shortlist]
    synthesizeShortlist --> uiResult["Shortlist + citations"]
```

| Node | Responsibility | LLM? | Fallback |
|------|----------------|------|----------|
| `parse_query` | Extract structured filters (skills, location, must-have, min experience) from free-text brief | Yes (JSON) | Treat full brief as a role keyword |
| `retrieve_candidates` | Parameterized SQL over `linkedin_api_profiles_parsed` with weighted lexical scoring | No | Auto-relax strict filters when zero rows return |
| `rank_candidates` | Per-candidate 5-dimension rubric (PhD/research, SF fit, technical, education, founder) + match reasons and risks | Yes | Deterministic per-dimension scorers over headline/about/experience/education JSON |
| `synthesize_shortlist` | Cited hiring-manager-ready summary of top candidates | Yes | Template list with `profile_id` citations |

Implemented in [`src/langgraph_app.py`](src/langgraph_app.py), with types in
[`src/schemas.py`](src/schemas.py) and DB access in
[`src/retriever.py`](src/retriever.py).

## Data

The demo reads from a MySQL table, `linkedin_api_profiles_parsed`, with the
following relevant columns:

- `profile_id` (unique)
- `name`, `headline`, `location`, `source_table`
- `about_text`, `about_char_count`
- `skills_json`, `skills_count`
- `experience_json`, `experience_count`
- `education_json`, `education_count`

The demo in this repo was built against a **1,000-row random sample** of
LinkedIn API profiles. You can point it at your own MySQL instance by
populating `.env` (see `.env.example`).

## Quickstart

Requires Python 3.9+ and access to the MySQL table above.

```bash
# 1. Clone + set up
git clone https://github.com/tafokints/langraph_ranker_sample.git
cd langraph_ranker_sample
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env with your DB_* values and OPENAI_API_KEY

# 3. Verify DB connectivity
python test_db_connection.py

# 4. Run the Streamlit demo
streamlit run app.py
```

Then open <http://localhost:8501>, type a role brief on the left, and click
**Run recruiter agent**.

### CLI mode (no UI)

```bash
python main.py "Senior technical recruiter hiring ML engineers" --top-k 6
```

### Smoke test (no UI)

```bash
python scripts/smoke_test.py
```
Runs three representative prompts (role-focused, skill-focused,
location-focused) and prints a PASS/FAIL report.

## Configuration

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | Yes | — | MySQL connection |
| `DB_PORT` | No | `3306` | MySQL port |
| `OPENAI_API_KEY` | Yes for LLM mode | — | Enables LLM parsing/ranking/synthesis |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Chat model used by all LLM nodes |

## Project layout

```
app.py                     Streamlit UI
main.py                    CLI entrypoint
test_db_connection.py      Minimal MySQL smoke test
requirements.txt
scripts/
  smoke_test.py            Headless end-to-end test across 3 prompts
src/
  __init__.py
  langgraph_app.py         LangGraph: 4 nodes + state + fallbacks
  retriever.py             Parameterized SQL with structured filters
  schemas.py               TypedDicts for graph state and parsed query
docs/
  streamlit_demo.png       Screenshot used in this README
.cursor/rules/
  karpathy-guidelines.mdc  Karpathy-inspired behavioral rules for agents
.streamlit/
  config.toml              Local theme + headless defaults
.env.example               Template for local env vars
```

## Ranking rubric (5-dimension scoring)

`rank_candidates` produces five sub-scores (0-10 each), grounded in structured
signals, then aggregates them into the overall `rank_score` using fixed weights
tuned for an SF-based startup founding/early-team lens:

| Dimension | Weight | Signals |
|-----------|--------|---------|
| Technical background | 0.30 | Technical titles and stack keywords (Python/PyTorch/AWS/...), skills count |
| Founder experience | 0.25 | Founder / co-founder / founding-engineer / CEO / CTO titles, YC / round / exited context |
| PhD / Researcher | 0.15 | PhD/doctorate, research roles (research scientist, postdoc, PI), publication venues (arXiv, NeurIPS, ...) |
| Education prestige | 0.15 | Top-tier or strong schools, highest degree bump |
| SF location fit | 0.15 | SF/Bay Area (10), CA tech (7), major US tech hub (5), US/remote (4), else (2), missing (1) |

A **deterministic feature extractor** always computes all 5 sub-scores from
`headline`, `about_text`, `experience_json`, `education_json`, and `location`.
When the LLM is enabled it receives those baselines in the prompt and is asked
to *refine* each sub-score (not start from zero), with a short evidence-grounded
reason per dimension. The final `rank_score` is always re-aggregated from the
merged sub-scores, so the overall number is consistent with its parts. If the
LLM call or JSON parse fails, the deterministic baseline is used unchanged.

Weights live in `DIMENSION_WEIGHTS` in [`src/langgraph_app.py`](src/langgraph_app.py),
resolved at import time by [`src/weights_loader.py`](src/weights_loader.py) which
reads [`config/weights.json`](config/weights.json) (and falls back to the
`DEFAULT_DIMENSION_WEIGHTS` constant if that file is missing or invalid).

## Calibration loop

The rubric self-improves by collecting per-candidate human labels and refitting
the five weights against them. There are two failure modes we calibrate against
separately:

- **Per-layer miscalibration** (a dimension's heuristic drifts from how a
  non-technical reviewer perceives it, e.g. we score "PhD" too generously).
- **Overall-score miscalibration** (individual dimensions are fine, but the
  weighted aggregate doesn't match perceived candidate fit).

### Label schema

Each label, written to the MySQL table `recruiter_rubric_labels` by
[`src/labels_store.py`](src/labels_store.py), is:

- 5 per-dimension scores (0-10, one per `DIMENSION_KEYS` entry)
- 1 overall score (0-10)
- optional short note
- labeler handle (`"me"` by default; multiple labelers are supported with no
  schema change)

Labels are captured inline from the Streamlit candidate card — expand
**Rate this candidate**, adjust sliders, click **Save rating**.

### Calibrator CLI

```bash
# Minimal report only; do not write weights.
python scripts/calibrate.py --dry-run

# Fit weights using all labels (requires >=15 by default).
python scripts/calibrate.py

# Fit only against your own labels.
python scripts/calibrate.py --labeler me --min-labels 20
```

For every run the CLI writes a markdown report to `reports/calibration_<timestamp>.md`
with:

1. Per-dimension **MAE**, signed **bias** (heuristic minus human), **Spearman**
   rank correlation, and heuristic/human std devs. Dimensions with
   `|bias| > 1.5` or `MAE > 2.0` are flagged as miscalibrated — fix the
   corresponding token list or base score in [`src/langgraph_app.py`](src/langgraph_app.py)
   and re-run the CLI to confirm the bias shrank.
2. Overall **MAE before** (current weights) vs **MAE after** (fitted weights).

Unless `--dry-run` is set and N >= `--min-labels`, a new `config/weights.json`
is written with version (`v1`, `v2`, ...), fit timestamp, and both MAEs. The
previous file is backed up to `config/weights.prev.json`.

### Weight fit

Given per-label heuristic dim-scores `x_k ∈ R^5` and human overall `y_k`:

\[
\min_{w \in \mathbb{R}^5} \sum_k (w \cdot x_k - y_k)^2
\quad\text{s.t.}\quad w_i \ge 0,\ \sum_i w_i = 1
\]

Implemented as `scipy.optimize.nnls` (non-negativity) followed by closed-form
Euclidean projection onto the probability simplex (Duchi et al. 2008) to
enforce `sum = 1`. Deleting `config/weights.json` reverts behavior to the
hardcoded defaults.

## Design notes and tradeoffs

- **Prototype retrieval.** Lexical SQL (`LIKE` with weighted hits) is used
  instead of embeddings/vector search. This keeps the demo minimal and
  explicit; the UI labels it as prototype retrieval so reviewers
  understand the scope.
- **Small sample.** The demo was exercised against a 1,000-row random
  sample. When a strict filter returns zero rows, the graph relaxes
  must-have/location/min-experience constraints once and retries — this is
  surfaced as a "Run warnings" message.
- **Error isolation.** Each node captures its own exceptions into
  `error_messages` so a failure at any step degrades gracefully rather
  than aborting the run.
- **No hidden state.** All shared data flows through a single `TypedDict`
  (`RecruiterGraphState`) — easy to inspect and test.

## License

MIT
