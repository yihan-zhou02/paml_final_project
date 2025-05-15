import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset, LogisticRegressionScratch
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

#############################################

st.title('Train Model')

#############################################
df = fetch_dataset()

def inspect_coefficients(models):
    st.write("Inspecting model coefficients (if applicable)...")
    for name, model in models.items():
        if name == 'Logistic Regression':
            st.write(f"{name} Coefficients: {model.weights}")
        else:
            st.write(f"{name} does not support coefficient inspection.")


if df is not None:
    st.dataframe(df)

    # Select variable to predict
    st.markdown('### Select variable to predict')
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=df.columns,
        key='feature_selectbox',
    )

    # Select input features
    st.markdown('### Select input features')
    feature_input_select = st.selectbox(
        label='Select features for classification input',
        options=df.columns,
        key='feature_select'
    )

    st.write(f'You selected input `{feature_input_select}` and output `{feature_predict_select}`.')

    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown('#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1
    )

    # Train models
    st.markdown('### Train models')
    model_options = ['Logistic Regression', 'Naive Bayes']

    model_select = st.multiselect(
        label='Select regression model for prediction',
        options=model_options,
    )

    st.write('You selected the following models: {}'.format(model_select))

    if st.button('Train Models'):
        # 如果在 Explore 页面中已经做过 TF-IDF，就用它；否则现场做
        if 'tfidf_matrix' not in st.session_state:
            # 自己在训练页重新生成 TF-IDF，但要保存 vectorizer
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vectorizer.fit_transform(df[feature_input_select].astype(str)).toarray()
            st.session_state["tfidf_matrix"] = X
            st.session_state["tfidf_vectorizer"] = vectorizer
        else:
            X = st.session_state["tfidf_matrix"]


        y = df[feature_predict_select].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number / 100.0, random_state=42)

        # 替换标签值：将正类 4 改为 1
        y_train = np.where(y_train == 4, 1, y_train)
        y_test = np.where(y_test == 4, 1, y_test)


        st.session_state["vectorizer"] = vectorizer
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        if 'Logistic Regression' in model_select:
            model_lr = LogisticRegressionScratch(learning_rate=0.1, epochs=300, reg_lambda=0.1, verbose_step=0)
            model_lr.fit(X_train, y_train)
            st.session_state["Logistic Regression"] = model_lr
            st.success("✅ Logistic Regression trained.")

        if 'Naive Bayes' in model_select:
            model_nb = GaussianNB()
            model_nb.fit(X_train, y_train)
            st.session_state["Naive Bayes"] = model_nb
            st.success("✅ Naive Bayes trained.")

    # Inspect coefficients
    st.markdown('### Inspect model coefficients')

    inspect_models = st.multiselect(
        label='Select models to inspect coefficients',
        options=model_select,
        key='inspect_multiselect'
    )

    models = {}
    for model_name in inspect_models:
        if model_name in st.session_state:
            models[model_name] = st.session_state[model_name]

    inspect_coefficients(models)

    st.write('Continue to Test Model')
else:
    st.warning("⚠️ Please upload a dataset.")
