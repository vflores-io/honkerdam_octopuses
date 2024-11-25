import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
import sys
import os
from gower import gower_matrix
# Add the project root to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.recommender import CustomGithubIssueRecommender

# tabs
st.set_page_config(page_title = 'Github Issue Recommender', page_icon = ':rocket:', layout = 'centered')

token = "***"
# Headers including the token for authentication 
headers = {"Authorization": f"token {token}"} 


# cache the NN recommender instance to avoid reloading on every interaction
@st.cache_resource
def load_recommender():
    model_path = 'backend/models/model.pth'
    tfdif_path = 'backend/models/tfidf_matrix.npz'
    df_path    = "data/intermediate_data/pr_df_clean_issues.parquet"
    return CustomGithubIssueRecommender(model_path, tfdif_path, df_path)

# initialize the recommender
recommender = load_recommender()

# Function to get GitHub user info
def get_github_user_info(username):
    # GitHub API URL for the user
    api_url = f"https://api.github.com/users/{username}" 
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        user_info = response.json()
        
        # Additional API requests for more information
        repos_url = user_info.get("repos_url")
        repos_response = requests.get(repos_url)
        total_commits = 0
        if repos_response.status_code == 200:
            repos = repos_response.json()
            for repo in repos:
                commits_url = f"{repo['url']}/commits"
                commits_response = requests.get(commits_url)
                if commits_response.status_code == 200:
                    total_commits += len(commits_response.json())

        return {
            "login": user_info.get("login"),
            "name": user_info.get("name"),
            "location": user_info.get("location"),
            "public_repos": user_info.get("public_repos"),
            "commits": total_commits,
            "public_gists": user_info.get("public_gists"),
            "followers": user_info.get("followers"),
            "following": user_info.get("following"),
            "bio": user_info.get("bio"),
            "created_at": user_info.get("created_at")
        }
    else:
        return {"error": f"Unable to fetch info for user {username}. Status code: {response.status_code}"}

# Function to find similar users using chunk processing
def find_similar_users(dataframe, user_info, k=10, chunk_size=1000):
    # Ensure the DataFrame is in the correct format
    df = dataframe.copy()
    
    # Convert 'created_at' to datetime and then to numerical age in days
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.tz_localize(None)
    df['account_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days
    
    # Features to use for comparison
    comparison_columns = ['public_repos', 'followers', 'commits', 'public_gists', 'following', 'account_age_days']
    
    # Drop the non-comparison columns for the comparison
    features_df = df[comparison_columns]
    
    # Prepare user info for comparison
    user_info_df = pd.DataFrame([user_info])
    user_info_df['created_at'] = pd.to_datetime(user_info_df['created_at'])
    user_info_df['created_at'] = user_info_df['created_at'].dt.tz_localize(None)
    user_info_df['account_age_days'] = (pd.Timestamp.now() - user_info_df['created_at']).dt.days
    user_features = user_info_df[comparison_columns]
    
    # Initialize lists to store distances and indices
    all_distances = []
    all_indices = []

    # Process data in chunks
    for i in range(0, len(features_df), chunk_size):
        chunk = features_df.iloc[i:i + chunk_size]
        combined = pd.concat([chunk, user_features], ignore_index=True)
        gower_distances = gower_matrix(combined)
        user_distances = gower_distances[-1, :-1]
        all_distances.extend(user_distances)
        all_indices.extend(range(i, min(i + chunk_size, len(features_df))))

    # Convert to numpy arrays
    all_distances = np.array(all_distances)
    all_indices = np.array(all_indices)

    # Find the nearest neighbors
    nearest_indices = np.argsort(all_distances)[:k]

    # Get the similar users
    similar_users = df.iloc[all_indices[nearest_indices]]
    
    # Return the similar users and their issues
    similar_user_issues = df.loc[df['id'].isin(similar_users['id']), ['repo_owner', 'repo_name', 'issue_title', 'mock_number']]
    return similar_users, similar_user_issues

# Function to get Good First Issues
def get_good_first_issues():
    url = "https://api.github.com/search/issues"
    params = {
        "q": "label:good-first-issue",
        "per_page": 100,
        "sort": "created",
        "order": "desc"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        issues = response.json()["items"]
        df = pd.DataFrame({
            "repo_owner": [issue["repository_url"].split("/")[-2] for issue in issues],
            "repo_name": [issue["repository_url"].split("/")[-1] for issue in issues],
            "issue_title": [issue["title"] for issue in issues],
            "created_at": [issue["created_at"] for issue in issues],
            "url": [issue["html_url"] for issue in issues]
        })
        return df
    else:
        st.error(f"Failed to fetch issues: {response.status_code}")
        return pd.DataFrame()

# Function to find similar Good First Issues
def find_similar_good_first_issues(good_first_issues_df, similar_user_issues, k=10):
    # Prepare the DataFrames for similarity comparison
    features_df = good_first_issues_df[['repo_owner', 'repo_name', 'issue_title']].copy()
    user_issues_df = similar_user_issues[['repo_owner', 'repo_name', 'issue_title']].copy()
        
    # Compute Gower similarity matrix
    combined_df = pd.concat([features_df, user_issues_df], ignore_index=True)
    gower_distances = gower_matrix(combined_df)
    
    # Extract the distances for the user issues (last rows)
    user_distances = gower_distances[-len(user_issues_df):, :-len(user_issues_df)]
    
    # Find the nearest neighbors
    nearest_indices = np.argsort(user_distances, axis=1)[:, :k]
    
    # Get the similar Good First Issues
    similar_good_first_issues = features_df.iloc[np.unique(nearest_indices.flatten())]
    
    return similar_good_first_issues



# Streamlit app
def run_user_profile_recommender():
    st.title("Good First Issues Recommender - Awesome App by 🐙Honkerdam Octopuses🐙")

    # Input for GitHub username with a unique key
    username = st.text_input("Enter GitHub username", key="username_input")

    # Slider for number of similar users
    k = st.slider("Number of similar users to find", min_value=1, max_value=20, value=10)

    if username:
        user_info = get_github_user_info(username)
        
        if "error" in user_info:
            st.error(user_info["error"])
        else:
            # Display user info
            st.write(f"**Username**: {user_info['login']}")
            st.write(f"**Name**: {user_info['name']}")
            st.write(f"**Location**: {user_info['location']}")
            st.write(f"**Number of Public Repos**: {user_info['public_repos']}")
            st.write(f"**Number of Commits**: {user_info['commits']}")
            st.write(f"**Number of Public Gists**: {user_info['public_gists']}")
            st.write(f"**Number of Followers**: {user_info['followers']}")
            st.write(f"**Number of Following**: {user_info['following']}")
            st.write(f"**Bio**: {user_info['bio']}")
            st.write(f"**Account Created At**: {user_info['created_at']}")
            
            # Load your dataset here
            # For demonstration purposes, we use a sample dataframe
            df= pd.read_csv("small_user_issues.csv")



            # Find similar users and their issues
            similar_users, similar_user_issues = find_similar_users(df, user_info, k=k)

            # Fetch Good First Issues
            good_first_issues_df = get_good_first_issues()

            # Find similar Good First Issues
            similar_good_first_issues = find_similar_good_first_issues(good_first_issues_df, similar_user_issues, k=k)

            # Display the results
            st.write(f"### The users most similar to {user_info['login']} are:")
            st.write(similar_users[['id', 'public_repos', 'followers', 'commits', 'public_gists', 'following', 'account_age_days']])
            
            st.write("### Their issues are:")
            st.write(similar_user_issues)
            
            st.write("### Good First Issues:")
            st.write(good_first_issues_df)

def run_keyword_based_recommender():
        # app title and description
    st.title('Good First Issues Recommender - Awesome App by 🐙Honkerdam Octopuses🐙')
    # st.markdown("""
    #             This app helps you find similar Github issues using a pre-trained neural network.
    #             Enter a keyworkd to search for issues, then select a specific issue to get recommendations.
    #             """)

    # user input for keyword search
    search_query = st.text_input("Search GitHub issues by keyword", "").lower()

    if search_query:
        # search for issues using the backend
        search_results = recommender.search_issues(search_query)

        if search_results:
            # display search results in a dropdown
            issue_options = {
                f'{issue['issue_title_clean']} (#{issue['issue_number']})': issue['index'] 
                for issue in search_results
            }
            selected_issue = st.selectbox(
                "Select an issue to find similar ones",
                options = list(issue_options.keys()),
            )

            # slider for number of recommendations
            num_recommendations = st.slider(
                "Number of recommendations to display",
                min_value = 1,
                max_value = 10,
                value = 5,
            )

            if selected_issue:
                # get recommendations
                issue_idx = issue_options[selected_issue]
                recommendations = recommender.recommend(issue_idx, num_recommendations)

                # display recommendations
                st.subheader("Recommended Issues")
                recommendation_table = {
                    "Repository": [recommender.pr_df.iloc[rec[0]]['repo_owner'] for rec in recommendations],
                    "Issue": [rec[2] for rec in recommendations],
                    "Similarity": [rec[1] for rec in recommendations],
                }
                st.dataframe(
                    recommendation_table,
                    use_container_width = True,
                )

                # plot similarity distribution
                fig = px.bar(
                    x = recommendation_table['Issue'],
                    y = recommendation_table['Similarity'],
                    labels = {
                        'x': 'Issue Title',
                        'y': 'Similarity Score',
                    },
                    title = 'Similarity Scores for Recommended Issues'
                )
                fig.update_layout(
                    xaxis_tickangle = -45,
                    height = 500,
                )
                st.plotly_chart(fig, use_container_width = True)

        else:
            st.info("No issues found for the given keyword. Try a different keyword.")



# tabs for different recommenders
tabs = st.tabs(['Profile-based Recommendations', 'Keyword-based Recommendations'])

with tabs[0]:
    run_user_profile_recommender()

with tabs[1]:
    run_keyword_based_recommender()
