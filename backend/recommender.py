import torch
import pandas as pd
import scipy.sparse as sp
import numpy as np

class SimilarityModel(torch.nn.Module):
    """
    neural network for predicting similarity between Github issues
    """
    def __init__(self, input_dim):
        super(SimilarityModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class CustomGithubIssueRecommender:
    def __init__(self, model_path, tfidf_path, df_path, device=None):
        """
        initialize the recommender with the trained model, tf-idf matrix and dataset

        parameters:
        - model_path: path to the trained model
        - tfidf_path: path to the serialized tf-idf matrix
        - df_path: path to the dataset (parquet format)
        - device: device to run the model on
        """

        # load the device
        self.device = device if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        # load the model
        self.tfidf_matrix = sp.load_npz(tfidf_path)
        self.model = self.load_model(model_path)
        self.pr_df = pd.read_parquet(df_path)

    def load_model(self, model_path):
        """
        load the pre-trained neural network model
        """

        # ensure model architecture matches
        input_dim = self.tfidf_matrix.shape[1] * 2
        model = SimilarityModel(input_dim)
        model.load_state_dict(torch.load(model_path, map_location = self.device))
        model.to(self.device)
        model.eval()

        return model
    
    def search_issues(self, keywords):
        """
        search for issues based on keywords
        """
        matches = self.pr_df[
            self.pr_df['issue_title_clean'].str.contains(keywords, case = False, na = False)
        ]
        return matches.reset_index().to_dict('records')
    
    def recommend(self, query_idx, top_n = 5):
        """
        recommend the most similar issues using the neural network model
        """

        query_vector = torch.tensor(
            self.tfidf_matrix[query_idx].toarray(),
            dtype = torch.float32
        ).flatten().to(self.device)

        similarities = []
        with torch.no_grad():
            for idx in range(self.tfidf_matrix.shape[0]):
                if idx != query_idx:
                    candidate_vector = torch.tensor(
                        self.tfidf_matrix[idx].toarray(),
                        dtype = torch.float32
                    ).flatten().to(self.device)

                    pair_vector = torch.cat([query_vector, candidate_vector], dim = 0).to(self.device)
                    pred_similarity = self.model(pair_vector.unsqueeze(0)).item()

                    similarities.append((idx, pred_similarity))
        
        # sort by similarity
        similarities = sorted(similarities, key = lambda x: x[1], reverse = True)

        # return top N recommendations
        return [
            (idx, score, self.pr_df.iloc[idx]['issue_title_clean'])
            for idx, score in similarities[:top_n]
        ]
    
if __name__ == '__main__':

    # paths to resources
    model_path = 'backend/models/model.pth'
    tfdif_path = 'backend/models/tfidf_matrix.npz'
    df_path    = "data/intermediate_data/pr_df_clean_issues.parquet"

    # initialize the recommender
    recommender = CustomGithubIssueRecommender(model_path, tfdif_path, df_path)

    # test search functionality
    print("searching for issues with keyword 'bug'")
    results = recommender.search_issues('bug')
    for res in results[:5]:
        print(f'{res["index"]}: {res["issue_title_clean"]}')

    # test recommendation functionality
    print("\ntop 5 recommendations for issue 73")
    recommendations = recommender.recommend(73)
    for idx, score, title in recommendations:
        print(f'index: {idx}, predicted similarity: {score:.2f}, title: {title}')