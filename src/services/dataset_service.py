import pickle
import pandas as pd
from src.services.diversity_service import calculate_diversity_score, get_wikirank_score


class DatasetService:
    def __init__(self, api):
        self.api = api

    def calculate_embeddings(self):
        for i in range(len(self.api.dataset)):
            content = self.api.dataset[i]['content']
            self.api.dataset[i]['embeddings'] = self.api.model.encode(content)
        print("Embeddings calculated")

    def save_dataset(self, pkl_path, scores_csv_path):
        self.calculate_embeddings()
        # Save dataset as a pickle file
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.api.dataset, f)

        print(f"Datasets saved as .pkl file at: {pkl_path}")

        # Calculate scores and save to CSV
        diversity_score = calculate_diversity_score(self.api.dataset)
        wikirank_score = get_wikirank_score(self.api.dataset, self.api.wikirank_df)
        final_score = (wikirank_score + 100 * diversity_score['Overall Diversity Score']) / 2

        scores = {
            "Dataset Size": len(self.api.dataset),
            "WikiRank Score": wikirank_score,
            "Diversity Score": diversity_score['Overall Diversity Score'],
            "Final Score": final_score
        }
        scores_df = pd.DataFrame([scores])
        scores_df.reset_index(inplace=True)
        scores_df.rename(columns={'index': 'id'}, inplace=True)
        scores_df.to_csv(scores_csv_path, index=False)
        print(f"Scores saved to CSV file at: {scores_csv_path}")
