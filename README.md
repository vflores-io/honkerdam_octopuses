# Build

## Team: Honkerdam Octopuses! 🚀🐙
We're on a mission to help **Novice Open Source Contributors** to overcome the overchoice issue, that might block them to do their First Contribution to OSS.
Here's an app that will help the **Novice Open Source Contributors** find their first Good First Issue, that's tailored to their profile.

---

## Project Title
Github Good First Issue Recommender System

---

## Description
This project is a web-based application designed to recommend similar GitHub issues using a simple neural network model, trained on commit data from GitHub. Its primary goal is to assist developers - especially those just starting out - in discovering **Good First Issues** based on their github users profile. By offering recommendations, the system makes it easier to navigate large repositories and helps **Novice Open Source Contributors** get started faster and more in line with their own strengths. 

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
 
![honkerdam_octopuses](https://github.com/user-attachments/assets/a7c7c2c8-141b-491d-b9bb-a86198c47a8a)


