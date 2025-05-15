# SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection

This project was developed for the **Practical Applications of Machine Learning (PAML)** course at Cornell Tech. It demonstrates how interpretable machine learning models can be used to classify sentiment in social media posts and uncover temporal trends using real-world data.

## 🔍 Project Overview

We implemented two baseline classification models:
- **Logistic Regression**
- **Naive Bayes**

Each model was evaluated with and without TF-IDF features. All models were implemented from scratch without using high-level machine learning libraries (e.g., no `scikit-learn` for training), and were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

The entire pipeline is deployed through an interactive **Streamlit** web app that allows users to upload data, train models, evaluate results, and make real-time predictions.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yihan-zhou02/sentitrend.git
cd sentitrend

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run streamlit_app/final_project.py
```