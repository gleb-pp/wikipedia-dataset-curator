# Wikipedia Dataset Curation Agent

Intelligent agent for collecting diverse and important Wikipedia pages under strict API constraints (6500 request limit).

## Overview

This project implements an autonomous agent that curates a high-quality dataset of 5,000 Wikipedia pages while maximizing lexical, categorical, and semantic diversity. The agent operates within tight API constraints and must make intelligent decisions about which pages to fetch without prior knowledge of content.

**Key challenges addressed:**
- Partial observability — no knowledge of page content before fetching
- Strict 6500 request limit for exploration and collection
- Must balance exploration (finding new topics) vs exploitation (retrieving quality pages)
- Cannot use WikiRank scores during selection (only for final evaluation)

## My Solution

### Core Architecture

The solution follows clean architecture principles with clear separation of concerns:

```
src/
├── domain/         # Core entities (Page with name, depth, embeddings)
├── agent/          # Decision-making logic (page selection, exploration strategy)
└── services/       # Wikipedia API wrapper, diversity calculations, persistence
```

### Key Technical Features

- **Embedding-based novelty tracking**: Uses SentenceTransformer + FAISS to avoid selecting semantically similar pages (prevents topic clustering)
- **Heuristic quality estimation**: Multi-factor scoring combining name relevance, page depth, and link popularity — all without accessing WikiRank
- **Multi-stage selection strategy**: Seed queries → breadth-first exploration → quality ranking
- **Link graph utilization**: Pages with more incoming links get higher priority
- **Smart filtering**: Excludes disambiguation pages, outlines, and lists automatically

### Algorithm

The agent's decision-making is driven by a weighted scoring function:

```
page_weight = name_rank × novelty × (1 / (1 + popularity))

where:
- name_rank: heuristic quality based on title characteristics
- novelty: 1 - max_similarity to previously selected pages (FAISS)
- popularity: number of times this page appeared in search results
```

### Performance

- Successfully collects 5000 pages within 6500 request limit
- Achieves high diversity scores across all three metrics
- Fully autonomous operation with no manual intervention

## Results

The agent produces:
- `dataset.pkl`: Complete dataset with page content, links, categories, and embeddings
- `scores.csv`: Diversity scores, WikiRank score, and final competition score

## Technology Stack

- **Python 3.14+**
- **SentenceTransformer** (all-MiniLM-L6-v2) — for semantic embeddings
- **FAISS** — efficient similarity search for novelty tracking
- **Wikipedia API** — data source
- **Poetry** — dependency management

## Project Structure

```
├── src/                    # Core implementation
│   ├── agent/             # PageVisitor with selection logic
│   ├── domain/            # Page entity with embeddings
│   └── services/          # API wrappers and utilities
├── en.tsv                 # WikiRank dataset (for evaluation only)
├── pyproject.toml         # Poetry dependencies
└── README.md
```

## Installation & Usage

```bash
# Install dependencies
poetry install

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the agent
poetry run python3 -m src.main
```

**Note:** Full competition rules and API specification available in the original assignment description (see `problem.md`).
