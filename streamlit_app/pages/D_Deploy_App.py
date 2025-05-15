import streamlit as st

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

#############################################

st.title('Deploy Application')

#############################################

# 检查是否已有训练好的模型和向量器
vectorizer = st.session_state.get("tfidf_vectorizer", None)
available_models = [key for key in st.session_state if key in ["Logistic Regression", "Naive Bayes"]]

if vectorizer and available_models:
    st.markdown("### Predict Sentiment from Your Own Input")
    
    model_select = st.selectbox("Select model for prediction", available_models)
    user_input = st.text_area("Enter your sentence:", "")

    if user_input and st.button("Predict"):
        # 进行向量化
        X_input = vectorizer.transform([user_input]).toarray()

        # 获取模型并预测
        model = st.session_state[model_select]
        prediction = model.predict(X_input)[0]

        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: **{label}**")
else:
    st.warning("⚠️ Please ensure you have trained a model and applied TF-IDF in earlier steps.")
