# Wikipedia Dataset Curation Agent - Problem Statement

## Description

In the age of AI and data-driven innovation, the quality and diversity of datasets hold the key to building robust, intelligent systems. Diverse datasets ensure that AI models are not biased, cover a wide range of scenarios, and generalize well to unseen data. This competition challenges you to leverage the Wikipedia API to build an agent capable of constructing a diverse and important dataset of 5,000 Wikipedia pages.

The problem lies not only in retrieving data but in making intelligent decisions. Every API request consumes valuable resources, and constraints like request limits force us to be selective. Your agent must decide which pages to retrieve, how to diversify the dataset efficiently, and how to adapt when faced with incomplete data. For example, should you prioritize highly linked pages or explore a variety of topics? How can you ensure semantically diverse content while respecting API constraints?

The goal is to demonstrate the power of automated decision-making in dataset curation, balancing constraints with the ultimate aim of diversity and importance.

**Deadline:** 30 NOV 23:59

## Task

Your task is to design and implement an agent to automatically collect a dataset of 5,000 Wikipedia pages. Your solution should aim to maximize diversity based on the following four metrics:

- **Lexical Diversity**: Measures variation in vocabulary across documents.
- **Categorical Diversity**: Captures the breadth of categories in the dataset.
- **Semantic Diversity**: Evaluates how distinct the content of each document is in terms of meaning.

These diversity metrics will be combined into a diversity score with different weights for each factor. Later, the WikiRank score (averaged across pages) will be added to calculate your final competition score.

## Rules and Constraints

### Allowed API Methods

| Method | Description |
|--------|-------------|
| `fetch_page(page_name)` | Retrieves a page if the agent already knows its name |
| `search_pages(query)` | Returns a list of page names relevant to a query (max 10 results) |
| `get_usage_summary()` | Returns summary of API usage (requests used, limit, known pages) |
| `save_page(page_name)` | Saves the specified page to the submission dataset |
| `save_dataset(pkl_path, scores_csv_path)` | Saves submission datasets to specified paths |
| `is_legal_page(page_name)` | Checks if it is legal to fetch a page with the given name |

### Usage Constraints

- Limited to **6,500 total calls** for `fetch_page()` and `search_pages()` combined
- Each call to either function increments the request counter

### Critical Rules

- Cannot retrieve a page with a name not previously encountered
- Cannot retrieve a page not in the legal pages list, even if the name is known
- Cannot retrieve category information directly from the API
- Cannot depend on WikiRank score to choose pages or guide searches
- Agent must run fully autonomously without manual intervention

## Evaluation

### Dataset Requirements

- Exactly 5,000 pages
- Save as CSV for Kaggle submission and PKL for Moodle
- Use `save_dataframe_to_csv(csv_path, pkl_path)` for final submission

### Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Lexical Diversity | 0.3 | Vocabulary variation across documents |
| Categories Diversity | 0.3 | Breadth of categories in the dataset |
| Semantic Diversity | 0.4 | Distinctness of content meaning |

**Diversity Score** is calculated using the provided Python script incorporating the three metrics above.

**WikiRank Score** is the average WikiRank of all pages in your dataset (evaluated after submission).

**Final Score Calculation:**
```
Final Score = (Diversity Score × 100 + Average WikiRank) / 2
```

## API Reference

### WikipediaAPI Class

```python
class WikipediaAPI:
    def __init__(self, page_request_limit=6500, wikirank_datasets_with_quality_scores_en_tsv='path/to/en.tsv'):
        """
        Initialize the API with request limit and WikiRank dataset.
        """
    
    def search_pages(self, query):
        """
        Search for Wikipedia pages by query.
        Returns up to 10 page names matching the query.
        Increments request counter.
        """
    
    def fetch_page(self, page_name):
        """
        Fetch a Wikipedia page by name.
        Page must be known and legal.
        Returns page info with title, content, url, links.
        Increments request counter.
        """
    
    def save_page(self, page_name):
        """
        Save a fetched page to the dataset.
        Page must exist in fetched_pages.
        """
    
    def save_dataset(self, pkl_path, scores_csv_path):
        """
        Calculate embeddings, save dataset as pickle,
        compute scores, and save to CSV.
        """
    
    def is_legal_page(self, page_name):
        """Check if page is in legal pages list."""
    
    def get_usage_summary(self):
        """Get current API usage statistics."""
```

### Diversity Calculation Functions

```python
def calculate_lexical_diversity(documents):
    """Calculate ratio of unique words to total words."""

def calculate_semantic_diversity(embeddings):
    """Calculate diversity based on embedding similarities (1 - avg cosine similarity)."""

def calculate_category_diversity(articles):
    """Calculate diversity based on category coverage and pairwise overlap."""

def calculate_diversity_score(submission, weights=None):
    """Calculate overall diversity score with weighted metrics."""
```

## Submission Format

### Required Deliverables

1. **Kaggle**: Single CSV file with the dataset
2. **Moodle**:
   - PKL file (format: `m-mousatat.pkl`)
   - IPYNB notebook with a small report about the approach

### Bonus Opportunity (+3 points)

Propose a better evaluation metric idea:
- Submit to [Google Sheet](https://docs.google.com/spreadsheets/d/1yUad5_CUu9zUVLT82_6M6x992yqxAahC_03kJy7aArw/edit?usp=sharing)
- Provide thorough explanation
- Points awarded only if the metric idea is approved

## Bonus Assignment: Beyond Bare Queries (25% Bonus)

### Task 1: Research and Analysis

Answer the following questions based on the "Beyond Bare Queries" paper **without using LLM or AI assistance**:

1. **Creative Applications**: What is the most creative application you see for this paper? Explain why it's valuable and provide specific use cases.

2. **Improvements**: How can this paper's approach be improved? Identify 2-3 specific areas (accuracy, efficiency, scalability, robustness) and explain current limitations, proposed improvements, and implementation challenges.

3. **Real-Time Dynamic Scenes**: How can this method work in real-time with dynamic scenes (motion environments like simulators, games, real-time video)? Address computational efficiency, temporal consistency, handling scene changes, and required optimizations.

4. **Fundamental Limitations**: What are the fundamental theoretical or architectural limitations of using 3D scene graphs for open-vocabulary object grounding? Consider:
   - How does the discrete graph structure limit handling of continuous spatial relationships or partial occlusions?
   - What happens when the scene graph representation doesn't capture semantic relationships needed for a query?
   - Are there inherent trade-offs between graph completeness and computational tractability?

5. **Ambiguity and Uncertainty**: The paper uses deductive reasoning with language models. How does the system handle ambiguous queries where multiple objects could match? What mechanisms exist for uncertainty quantification, and how would you design a system that explicitly models and communicates its confidence?

6. **Generalization and Domain Transfer**: How well would this approach generalize to entirely new scene types, different sensor modalities, or scenes with different cultural assumptions?

### Task 2: Implementation

- Set up and run the paper's code on a custom sample NOT in the benchmark dataset
- Document: sample choice, setup steps, modifications needed, and results with screenshots

### Task 3: Video Documentation

Create a 30-60 second video covering:
- Implementation demo showing custom sample and results (2-3 minutes)

### Task 4: Submission

**Deliverables:**
- **PDF**: Answers to all questions, implementation details, results, and screenshots (named: `LastName_FirstName_BeyondBareQueries_Assignment.pdf`)
- **Video**: Upload link or file (ensure accessibility)

**Important:** No LLM/AI assistance for analysis questions. Use your own understanding. Custom sample must be original, not from the benchmark.

## Installation

```bash
pip install wikipedia sentence-transformers pandas numpy nltk scikit-learn scipy
```

## Grading Scheme

| Performance | Grade |
|-------------|-------|
| Top 20% of scores | 100% |
| Top 20-40% of scores | 90% |
| Top 40-70% of scores | 70% |
| Bottom 30% of scores | 50% |
| No submission | 0% |

## How to Avoid Getting 0

- Do not submit modified CSV files to Kaggle (destroys leaderboard)
- Do not submit CSV with dataset size less than required (5000 pages)
- Do not submit brute-force or random retrieval solutions
- Do not change the scoring metric
- Do not cheat or use WikiRank dataset in your agent logic (reserved for scoring only)

**Note:** All embeddings and scores will be recalculated to validate Kaggle leaderboard.

## References

mousatat. Information Retrieval Agent 2025. https://kaggle.com/competitions/information-retrieval-agent-2025, 2025. Kaggle.
