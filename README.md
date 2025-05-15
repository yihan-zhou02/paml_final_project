# SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection

This project was developed for the **Practical Applications of Machine Learning (PAML)** course at Cornell Tech. It demonstrates how interpretable machine learning models can be used to classify sentiment in social media posts and uncover temporal trends using real-world data.

Our teammembers are: Sylvia Li, Yifei Hu, Xavier Yin and Yihan Zhou.

## 📂 Dataset Note
Due to GitHub's file size limitation (100 MB), we use Git LFS to include the full Sentiment140 dataset in this repository.

So you need to first initialize Git LFS in your local repository:
```bash
git lfs install
```
Then, you can clone the repository and download the dataset.

You can also download the complete dataset (1.6 million tweets) from the official Kaggle page:

🔗 [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

This repository also includes a randomly sampled subset of **20,000 tweets**, stored in `sentiment140_sampled_20000.csv`, which we used for model training, evaluation, and demonstration in the Streamlit application.

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
git clone https://github.com/yihan-zhou02/paml_final_project.git
cd sentitrend
```
### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run streamlit_app/final_project.py
```