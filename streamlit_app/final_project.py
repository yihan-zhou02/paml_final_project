import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

#############################################

description = """ This project explores the use of machine learning models to classify sentiment in social media posts and uncover temporal trends. Using the Sentiment140 dataset, we trained and evaluated two interpretable models — logistic regression and Naive Bayes — each with and without TF-IDF feature representations.

Our pipeline includes custom model implementations, manual hyperparameter tuning, evaluation with multiple metrics (accuracy, precision, recall, F1 score, and AUC-ROC), and a Streamlit-based interface for testing and deployment.

**Team Members**:
- Sylvia Li 
- Yifei Hu
- Xavier Yin
- Yihan Zhou

We hope this tool provides a clear demonstration of how interpretable ML techniques can support social media sentiment monitoring."""

st.markdown(description)

st.markdown("Click **Explore Dataset** to get started.")
