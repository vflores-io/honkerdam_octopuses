# Build

## Team: Honkerdam Octopuses! 🚀🐙

## Project Title
Github Good First Issue Recommender System

## Description
This project is a web-based application designed to recommend similar GitHub issues using a simple neural network model, trained on commit data from GitHub. Its primary goal is to assist developers - especially those just starting out - in discovering **good first issues** based on their queries, which would ideally be based on their initial interests. By offering recommendations, the system makes it easier to navigate large repositories and helps **new contributors** get started faster and more in line with their own strengths. 

The app combines TF-IDF-based text processing with machine learning to provide an accessible solution.

## Core Features
- **Neural Network-Based Recommendations**: A lightweight PyTorch model predicts issue similarity based on textual descriptions.
- **TF-IDF Integration**: Preprocessed issue text to generate meaningful embeddings for similarity calculations.
- **Interactive Web Application**: Built with Streamlit, allowing users to search for issues by keywords, select an issue, and view recommended related issues.
- **Visualization**: Recommendations are displayed in a user-friendly table with an optional bar chart for similarity scores.
