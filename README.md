# Build

## Team: Honkerdam Octopuses! 🚀🐙
We're a team passionate about data and simplifying workflows for developers. We're focusing on making it easier to get into the data world and improving onboarding for open-source newcoming contributors.

---

## Project Title
Github Good First Issue Recommender System

---

## Description
This project is a web-based application designed to recommend similar GitHub issues using a simple neural network model, trained on commit data from GitHub. Its primary goal is to assist developers - especially those just starting out - in discovering **good first issues** based on their queries, which would ideally be based on their initial interests. By offering recommendations, the system makes it easier to navigate large repositories and helps **new contributors** get started faster and more in line with their own strengths. 

The app combines TF-IDF-based text processing with machine learning to provide an accessible solution.

---

## **Core Features**
- **Neural Network-Based Recommendations**: A PyTorch-based model predicts issue similarity based on textual descriptions.
- **TF-IDF Integration**: Text preprocessing using TF-IDF vectors creates a meaningful representation of issue content for similarity calculations.
- **Interactive Web Application**:
  - Built using Streamlit for an intuitive user interface.
  - Allows users to search for issues by keywords, select an issue of interest, and view tailored recommendations.
- **Visualization**:
  - Displays recommendations in a clean and user-friendly table.
  - Optional bar chart to show similarity scores visually.

  ---

  ## **Submission Items**

  ### **Deliverable**
  - **Web Application**:
    - Build with Streamlit and integrated with a pre-trained PyTorch model.
    - Fully functional, enabling keyword-based search, issue selection, and recommendation generation.
    - Provides an interactive, user-friendly interface.

### **Documentation**
1. **App Architecture**:
    - Explanation of the recommender system workflow via notebooks and well-commented scripts, from data preprocessing to model inference and frontend integration.
    - Justification for key design choices, including the use of a simple neural network and TF-IDF-based feature extraction.
2. **Core Features**:
    - Detailed descriptions of the recommender's functionalities and how they were implemented.
    - Highlights of the Streamlit interface and interactive features.

---

## **Additional Notes**
- The recommender system was developed with hackathon constraints in mind, focusing on building a scalable prototype over perfec accuracy.
- Feature improvements include:
    - Training on larger and more ad-hoc datasets.
    - Incorporating contextual embeddings (e.g. BERT).
    - Enhancing the user experience with additional visualizations and filtering options.

