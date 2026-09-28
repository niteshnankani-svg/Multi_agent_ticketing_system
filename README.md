# Multi-Agent Ticketing System

A LangGraph-based support ticket router. An incoming ticket is cleaned, checked against a semantic cache, classified into one of four categories by TypeSafe's Jev model (via a `Choice` question), and handed off to a category-specific agent (finance, backend, internal, or general) to draft an answer, with a human-escalation path for low-confidence cases. An earlier fine-tuned local classifier (`models/router`, see `train_router.py`) is no longer used at runtime but is kept as a training artifact/reference.

## Repository layout

- `graph.py`, `state.py` — LangGraph pipeline definition and shared ticket state
- `ingest.py`, `semantic_cache.py` — ticket preprocessing and cache lookup
- `generate_answer*.py`, `internal_agent.py` — per-category answer-generation agents
- `build_kb*.py`, `expand_kb.py` — knowledge-base construction for retrieval (backed by `stores/` ChromaDB, gitignored)
- `train_router.py`, `split_data.py`, `merge_datasets.py`, `map_categories.py` — router classifier training and dataset prep
- `models/` — trained classifier checkpoints (gitignored, not versioned)
- `runtime/` — SQLite checkpoint DB for LangGraph state (gitignored)
- `data/raw/` — source and derived datasets (tracked via Git LFS, see below)

## Datasets (`data/raw/`)

The router is trained to classify tickets into four categories — **backend**, **finance**, **general**, **internal** — merged from three upstream sources: an IT service-desk dataset (`it_service`), a bilingual customer-support ticket dataset (`bueck`), and the BANKING77 intent dataset (`banking77`).

| File | Rows | Columns | Description |
|---|---|---|---|
| `all_categorized.csv` | 69,758 | `text, category, source` | Full merged pool of all categorized tickets (all three sources combined) prior to the train/val/test split. |
| `train.csv` | 48,830 | `text, category, source` | Training split of the merged, categorized dataset. |
| `val.csv` | 10,464 | `text, category, source` | Validation split. |
| `test.csv` | 10,464 | `text, category, source` | Test split. |
| `train_boosted.csv` | 49,914 | `text, category, source` | Training split with class-balancing/augmentation applied (minority-category oversampling). |
| `train_boosted2.csv` | 49,280 | `text, category, source` | Second boosted-training variant, used for router training experiments. |
| `train_with_answers.csv` | 48,830 | `text, category, source, answer` | Same rows as `train.csv`, with reference answers attached for answer-generation training/eval. |
| `tickets.csv` | 20,000 | `subject, body, answer, type, queue, priority, language, tag_1–tag_8` | Raw bilingual (English/German) customer support ticket dataset ("bueck" source), with ground-truth queue, priority, and tags. |
| `tickets_en.csv` | 11,923 | same as `tickets.csv` | English-only subset of `tickets.csv`. |
| `tickets_categorized.csv` | 11,922 | `tickets_en.csv` columns + `category, text` | `tickets_en.csv` mapped into the 4-class category scheme via `map_categories.py`. |
| `it_service.csv` | 47,837 | `Document, Topic_group` | IT service-desk ticket dataset; `Topic_group` includes Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights. Identical content to `all_tickets_processed_improved_v3.csv`. |
| `all_tickets_processed_improved_v3.csv` | 47,837 | `Document, Topic_group` | Duplicate copy of `it_service.csv` (kept for source-file provenance). |
| `banking77_train.csv` | 10,003 | `text, category` | BANKING77 training split — 77 fine-grained banking intents, used as the `finance` source. |
| `banking77_test.csv` | 3,080 | `text, category` | BANKING77 test split. |
| `it_service/sales_data.csv` | 1,000 | `Product_ID, Sale_Date, Sales_Rep, Region, Sales_Amount, ...` | Synthetic sales transaction data, unrelated to ticket categorization — used for a separate demo/agent scenario. |

### Git LFS

CSVs under `data/raw/` are tracked with [Git LFS](https://git-lfs.github.com) (see `.gitattributes`). After cloning, run:

```bash
git lfs pull
```

to fetch the actual file contents (a plain `git clone` only fetches LFS pointer files).

## Getting started

Model checkpoints (`models/`), the vector store (`stores/`), and the LangGraph checkpoint DB (`runtime/`) are gitignored and must be built locally — see `train_router.py` and `build_kb.py` to regenerate them from the datasets above.
