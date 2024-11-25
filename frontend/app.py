import streamlit as st
import plotly.express as px
import sys
import os

# Add the project root to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from backend.recommender import CustomGithubIssueRecommender

# set up page configuration
st.set_page_config(page_title = 'Github Issue Recommender', page_icon = ':rocket:', layout = 'centered')

# cache the recommender instance to avoid reloading on every interaction
@st.cache_resource
def load_recommender():
    model_path = 'backend/models/model.pth'
    tfdif_path = 'backend/models/tfidf_matrix.npz'
    df_path    = "data/intermediate_data/pr_df_clean_issues.parquet"
    return CustomGithubIssueRecommender(model_path, tfdif_path, df_path)

# initialize the recommender
recommender = load_recommender()

# app title and description
st.title('Github Issue Recommender Super Awesome App')
st.markdown("""
            This app helps you find similar Github issues using a pre-trained neural network.
            Enter a keyworkd to search for issues, then select a specific issue to get recommendations.
            """)

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
                "Repository": [recommender.pr_df.iloc[rec[0]]['repo'] for rec in recommendations],
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