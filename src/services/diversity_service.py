import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# nltk.download('punkt_tab')
# nltk.download('punkt')


def preprocess_documents(documents):
    """Ensure all documents are valid strings and handle invalid data."""
    if not isinstance(documents, list):
        raise ValueError("Input documents must be a list.")
    return [doc if isinstance(doc, str) else "" for doc in documents]


def calculate_lexical_diversity(documents):
    """
    Calculate the lexical diversity of a collection of documents.

    Args:
        documents (list of str): The list of documents (strings) to be analyzed.

    Returns:
        float: The ratio of unique words to total words in the documents.
    """
    total_words = sum(len(nltk.word_tokenize(doc)) for doc in documents if doc.strip())
    unique_words = len(set(word.lower() for doc in documents for word in nltk.word_tokenize(doc) if doc.strip()))
    return unique_words / total_words if total_words > 0 else 0


def calculate_semantic_diversity(embeddings):
    if len(embeddings) < 2:
        return 0
    cosine_sim = cosine_similarity(embeddings)
    avg_similarity = np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
    return 1 - avg_similarity


def calculate_category_diversity(articles):
    """
    Calculate diversity based on the intersection and coverage of categories in the dataset.

    Args:
        articles_str (list of str): List of articles, each represented by a string of categories.

    Returns:
        dict: A dictionary with 'Coverage', 'Overlap', and 'Category Diversity Score'.
    """
    # Coverage: Average number of categories per article
    avg_coverage = np.mean([len(categories) for categories in articles])

    # Pairwise Overlap Calculation: Measures the overlap of categories between pairs of articles
    overlap_sum = 0
    article_pairs = list(combinations(articles, 2))
    for cat1, cat2 in article_pairs:
        set1, set2 = set(cat1), set(cat2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        overlap_sum += intersection / union if union else 0

    # Global Diversity: Measures how diverse the dataset is based on category overlap
    max_possible_overlap = len(article_pairs)
    diversity_score = 1 - (overlap_sum / max_possible_overlap) if max_possible_overlap > 0 else 0
    return {
        'Average Coverage': avg_coverage,
        'Category Overlap Sum': overlap_sum,
        'Category Diversity Score': diversity_score
    }


def get_dataset_column(dataset, columns_name):
    column = []
    for i in dataset:
        column.append(i[columns_name])
    return column


def calculate_diversity_score(submission, weights=None):
    """
    Calculate an overall diversity score by combining lexical, semantic, and category-based diversity.

    Args:
        submission (pd.DataFrame): The submission data containing content and categories.
        weights (dict, optional): Weights for lexical, semantic, and category diversities.

    Returns:
        dict: A dictionary with the individual diversity scores and the overall diversity score.
    """
    documents = get_dataset_column(submission, 'content')
    categories = get_dataset_column(submission, 'categories')
    embeddings = get_dataset_column(submission, 'embeddings')
    if weights is None:
        weights = {'lexical': 0.3, 'semantic': 0.4, 'category': 0.3}

    # Lexical Diversity: Calculated based on word diversity in the documents
    try:
        lexical_diversity = calculate_lexical_diversity(documents)
    except Exception as e:
        print(f"Error in lexical diversity calculation: {e}")
        lexical_diversity = 0

    # Semantic Diversity: Extracted from the 'similarity_score' of the submission
    try:
        semantic_diversity = calculate_semantic_diversity(embeddings)
    except Exception as e:
        print(f"Error in semantic diversity calculation: {e}")
        semantic_diversity = 0

    # Category Diversity: Calculated based on the diversity of categories across the dataset
    try:
        category_diversity = calculate_category_diversity(categories)['Category Diversity Score']
    except Exception as e:
        print(f"Error in semantic diversity calculation: {e}")
        category_diversity = 0

    # Calculate overall diversity score as a weighted sum of individual scores
    diversity_score = (
        weights['lexical'] * lexical_diversity +
        weights['semantic'] * semantic_diversity +
        weights['category'] * category_diversity
    )

    return {
        'Lexical Diversity': lexical_diversity,
        'Semantic Diversity': semantic_diversity,
        'Category Diversity': category_diversity,
        'Overall Diversity Score': diversity_score
    }


def get_wikirank_score(dataset, wikirank_df):
    """
    Calculate the mean WikiRank score for a given dataset, ensuring all titles are present in the WikiRank dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing a 'title' column.
        wikirank_df (pd.DataFrame): The DataFrame containing 'page_name' and 'wikirank_quality'.

    Returns:
        float: The mean WikiRank score for the dataset.

    Raises:
        ValueError: If any titles in the dataset are not found in wikirank_df['page_name'].
    """
    dataset = pd.DataFrame({'title': get_dataset_column(dataset, 'title')})
    # Check if all titles in the dataset are present in the WikiRank dataset
    missing_titles = set(dataset['title']) - set(wikirank_df['page_name'])
    if missing_titles:
        raise ValueError(f"The following titles are missing from wikirank_df['page_name']: {missing_titles}")

    # Merge the datasets to calculate the mean WikiRank score
    merged_df = dataset.merge(wikirank_df, left_on='title', right_on='page_name', how='inner')

    # Extract the WikiRank quality scores and calculate the mean score
    scores = merged_df['wikirank_quality']
    return scores.mean()
