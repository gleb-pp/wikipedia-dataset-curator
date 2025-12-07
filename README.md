# Wikipedia Dataset Curation Agent

## Assignment Description

In the age of AI and data-driven innovation, the quality and diversity of datasets hold the key to building robust, intelligent systems. Diverse datasets ensure that AI models are not biased, cover a wide range of scenarios, and generalize well to unseen data. This competition challenges you to leverage the Wikipedia API to build an agent capable of constructing a diverse and important dataset of 5,000 Wikipedia pages.

The problem lies not only in retrieving data but in making intelligent decisions. Every API request consumes valuable resources, and constraints like request limits force us to be selective. Your agent must decide which pages to retrieve, how to diversify the dataset efficiently, and how to adapt when faced with incomplete data. For example, should you prioritize highly linked pages or explore a variety of topics? How can you ensure semantically diverse content while respecting API constraints?

The goal is to demonstrate the power of automated decision-making in dataset curation, balancing constraints with the ultimate aim of diversity and importance.

## Task

Your task is to design and implement an agent to automatically collect a dataset of 5,000 Wikipedia pages. Your solution should aim to maximize diversity based on the following four metrics:

- **Lexical Diversity**: Measures variation in vocabulary across documents.
- **Categorical Diversity**: Captures the breadth of categories in the dataset.
- **Semantic Diversity**: Evaluates how distinct the content of each document is in terms of meaning.

These diversity metrics will be combined into a diversity score with different weights for each factor. Later, the WikiRank score (averaged across pages) will be added to calculate your final competition score.

## Rules and Constraints

### Allowed API Methods
- `fetch_page(page_name)`: Retrieves a page if the agent already knows its name.
- `search_pages(query)`: Returns a list of page names relevant to a query.
- `get_usage_summary()`: Returns a summary of API usage, including the number of requests used, the request limit, and the list of known pages.
- `save_page(page_name)`: Save the page with specific page_name to the submission dataset.
- `save_dataset(pkl_path, scores_csv_path)`: save the submission datasets with specific paths.
- `is_legal_page(page_name)`: check if it is legal to fetch a page with name page_name or not.

### Usage Constraints
Each time agent call `fetch_page(page_name)` or `search_pages(query)`, the counter will track how many times agent are retrieving pages. You have a limited amount of **6500 calls** for both functions.

### Rules
- Agent are not allowed to retrieve a page with page name that you did not pass through before.
- Agent are not allowed to retrieve a page with page name that is not in the legal pages list, even if the agent know it's name.
- Agent are not allowed to retrieve categories info from the API.
- Agent are not allowed to depend on WikiRank score to choose the pages or search for pages.
- The agent must run fully autonomously, without manual intervention.

## Evaluation

### Dataset Requirements
The dataset must contain exactly 5,000 pages.

### Request Limitations
You are limited to 6,500 calls to `search_pages(query)` and `fetch_page(page_name)`.

### Diversity Objective
Your dataset should be as diverse as possible according to the given diversity metrics with the following weights:
- Lexical Diversity: 0.3
- Categories Diversity: 0.3
- Semantic Diversity: 0.4

### Importance Objective
Your dataset should be as important as possible according to the WikiRank score.

### Evaluation Criteria
**Diversity Score** is calculated using the provided Python script and incorporates the three diversity metrics.

**WikiRank Score** is the average WikiRank of all pages in your dataset.

**Final Score Calculation**:
```
Final Score = (Diversity Score × 100 + Average WikiRank) / 2
```

## Architecture

The project is structured following clean architecture principles with separation of concerns:

### Domain Layer (`src/domain/page.py`)
Contains the core business entity `Page` which encapsulates:
- Page name and exploration depth
- Content storage
- Name embedding generation using SentenceTransformer
- Quality estimation heuristics (name-based rank and content-based rank)

### Agent Layer (`src/agent/page_visitor.py`)
Implements the decision-making logic:
- Seed query management from a curated list of starting topics
- Page selection algorithm based on weighted scoring (name rank × novelty / popularity)
- Novelty tracking using FAISS with normalized embeddings
- Link graph processing and depth control
- Final ranking of collected pages

### Services Layer
- **Wikipedia Service** (`src/services/wikipedia_service.py`): Wrapper around the official WikipediaAPI with additional error handling
- **Diversity Service** (`src/services/diversity_service.py`): Implementation of lexical, semantic, and category diversity calculations
- **Dataset Service** (`src/services/dataset_service.py`): Handles dataset persistence and scoring

### Key Design Decisions

- **Multi-stage selection**: Initial seed queries → breadth-first exploration → quality ranking
- **Novelty tracking**: Using FAISS with normalized embeddings to avoid selecting semantically similar pages
- **Quality estimation**: Heuristic scoring without relying on WikiRank (complies with competition rules)
- **Link graph utilization**: Pages with more incoming links get higher priority
- **Depth control**: Limits exploration depth to 3 to balance breadth vs. depth
- **Filtering**: Excludes disambiguation pages, outlines, and lists to focus on substantive content

## Installation

```bash
# Clone the repository
git clone https://github.com/gleb-pp/wikipedia-dataset-curator
cd wikipedia-dataset-curator

# Install dependencies using Poetry
poetry install

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Place the WikiRank dataset file (en.tsv) in the project root
```

## Usage

### Running the Agent

```bash
# Activate the virtual environment
poetry shell

# Run the dataset curation agent
python3 -m src.main
```

The agent will:
1. Initialize with seed queries from a curated list
2. Explore Wikipedia pages while respecting API limits
3. Select and retrieve pages based on quality heuristics
4. Save the final dataset and calculate scores

### Output Files

After execution, the following files are generated:
- `dataset.pkl`: Pickle file containing the collected pages with content, links, categories, and embeddings
- `scores.csv`: CSV file with diversity score, WikiRank score, and final score

## Project Structure

```
.
├── en.tsv                    # WikiRank dataset
├── poetry.lock               # Poetry lock file
├── pyproject.toml            # Project dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── agent/
    │   └── page_visitor.py   # Main agent logic
    ├── domain/
    │   └── page.py           # Page entity
    ├── main.py               # Entry point
    └── services/
        ├── dataset_service.py   # Dataset persistence
        ├── diversity_service.py # Diversity calculations
        └── wikipedia_service.py # Wikipedia API wrapper
```

## Dependencies

- Python 3.14+
- wikipedia
- sentence-transformers
- faiss-cpu
- pandas
- numpy
- nltk
- scikit-learn

## Academic Integrity

- Plagiarism detection is enabled
- Use of language models (LLMs) is monitored
- Sharing solutions or collaboration on queries will result in penalties
