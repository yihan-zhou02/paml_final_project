import streamlit as st                  # pip install streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import fetch_dataset
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################

# Initialize session state for storing processed data
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Load and display initial data
df = fetch_dataset()

if df is not None:
    # Display original dataframe
    st.markdown('### Initial Dataset Overview')
    st.write(f"Dataset Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Basic statistics
    st.markdown('### Basic Statistics')
    st.write(df.describe())
    
    # Check for missing values
    st.markdown('### Missing Values Analysis')
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    st.write(missing_data)
    
    # Handle missing values
    st.markdown('### Handle Missing Values')
    if st.checkbox('Remove rows with missing values'):
        df = df.dropna()
        st.write(f"Shape after removing missing values: {df.shape}")
    
    # # Text preprocessing
    # st.markdown('### Text Preprocessing')
    
    # # def preprocess_text(text):
    # #     if isinstance(text, str):
    # #         # Convert to lowercase
    # #         text = text.lower()
    # #         # Remove special characters and digits
    # #         text = re.sub(r'[^a-zA-Z\s]', '', text)
    # #         # Tokenize
    # #         tokens = word_tokenize(text)
    # #         # Remove stopwords
    # #         stop_words = set(stopwords.words('english'))
    # #         tokens = [t for t in tokens if t not in stop_words]
    # #         # Lemmatization
    # #         lemmatizer = WordNetLemmatizer()
    # #         tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # #         return ' '.join(tokens)
    # #     return text
    
    
    # import re

    # def preprocess_text(text):
    #     # Convert to lowercase
    #     text = text.lower()
    #     # Remove URLs, mentions, non-alphabetic chars
    #     text = re.sub(r"http\S+|@\S+|[^a-z\s]", "", text)
    #     # Split by space
    #     tokens = text.split()
    #     # Simple stopword list
    #     stopwords = set(["the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "that", "this"])
    #     tokens = [word for word in tokens if word not in stopwords]
    #     return " ".join(tokens)
    
    # if st.checkbox('Preprocess text data'):
    #     text_columns = df.select_dtypes(include=['object']).columns
    #     for col in text_columns:
    #         df[col] = df[col].apply(preprocess_text)
    #     st.success("✅ Text preprocessing completed!")
    #     st.dataframe(df)

    #     # TF-IDF transformation (optional)
    #     if st.checkbox('Apply TF-IDF transformation'):
    #         tfidf_column = st.selectbox("Select column to apply TF-IDF on", text_columns)
    #         vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    #         tfidf_matrix = vectorizer.fit_transform(df[tfidf_column].astype(str)).toarray()

    #         # Save both vectorizer and transformed matrix to session_state
    #         st.session_state["tfidf_vectorizer"] = vectorizer
    #         st.session_state["tfidf_matrix"] = tfidf_matrix

    #         st.success("✅ TF-IDF transformation applied and stored.")
    
    # Feature selection
    st.markdown('### Feature Selection')
    if st.checkbox('Remove irrelevant features'):
        # Add your feature selection logic here
        # For example, removing columns with high correlation
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)
        
        # Remove highly correlated features
        threshold = st.slider('Correlation threshold', 0.0, 1.0, 0.8)
        columns_to_drop = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    columns_to_drop.append(correlation_matrix.columns[i])
        df = df.drop(columns=set(columns_to_drop))
        st.write(f"Removed {len(set(columns_to_drop))} highly correlated features")
    
    # Outlier detection and removal
    st.markdown('### Outlier Detection and Removal')
    if st.checkbox('Remove outliers'):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        st.write(f"Shape after removing outliers: {df.shape}")
    
    # # Data normalization
    # st.markdown('### Data Normalization')
    # if st.checkbox('Normalize numerical features'):
    #     numeric_columns = df.select_dtypes(include=[np.number]).columns
    #     scaler = StandardScaler()
    #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    #     st.write("Numerical features have been normalized!")
    
    # Save processed data
    st.session_state.processed_df = df
    
    # Display final processed dataset
    st.markdown('### Final Processed Dataset')
    st.dataframe(df.head())
    
    # Display some visualizations
    st.markdown('### Data Visualizations')
    
    # Distribution of numerical features
    if st.checkbox('Show feature distributions'):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        selected_column = st.selectbox('Select feature to visualize', numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=selected_column, ax=ax)
        st.pyplot(fig)
    
    # Correlation heatmap
    if st.checkbox('Show correlation heatmap'):
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    st.markdown('### Continue to Train Model')
else:
    st.error('Failed to load the dataset. Please check the data source.')
