# import streamlit as st                  # pip install streamlit
# from helper_functions import fetch_dataset

# #############################################

# st.markdown("# Practical Applications of Machine Learning (PAML)")

# #############################################

# st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

# #############################################

# st.title('Test Model')

# #############################################

# df = None
# df = fetch_dataset()

# if df is not None:
#     st.markdown("### Get Performance Metrics")
#     metric_options = ['placeholder']

#     model_options = ['Logistic Regression', 'Naive Bayes']
    
#     trained_models = [model for model in model_options if model in st.session_state]

#     # Select a trained classification model for evaluation
#     model_select = st.multiselect(
#         label='Select trained models for evaluation',
#         options=trained_models
#     )

#     if (model_select):
#         st.write(
#             'You selected the following models for evaluation: {}'.format(model_select))

#         eval_button = st.button('Evaluate your selected classification models')

#         if eval_button:
#             st.session_state['eval_button_clicked'] = eval_button

#         if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
#             st.markdown('### Review Model Performance')

#             review_options = ['plot', 'metrics']

#             review_plot = st.multiselect(
#                 label='Select plot option(s)',
#                 options=review_options
#             )

#             if 'plot' in review_plot:
#                 pass

#             if 'metrics' in review_plot:
#                 pass

#     # Select a model to deploy from the trained models
#     st.markdown("### Choose your Deployment Model")
#     model_select = st.selectbox(
#         label='Select the model you want to deploy',
#         options=trained_models,
#     )

#     if (model_select):
#         st.write('You selected the model: {}'.format(model_select))
#         if('deploy_model' in st.session_state):
#             st.session_state['deploy_model'] = st.session_state[model_select]

#     st.write('Continue to Deploy Model')
import streamlit as st                  # pip install streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - SentiTrend: A Machine Learning Approach for Social Media Sentiment Analysis and Temporal Pattern Detection")

#############################################

st.title('Test Model')

#############################################
X_test = st.session_state.get("X_test", None)
y_test = st.session_state.get("y_test", None)

model_options = [key for key in st.session_state.keys() if key in ["Logistic Regression", "Naive Bayes"]]

if X_test is None or y_test is None:
    st.warning("⚠️ You need to train a model first before testing.")
else:
    st.markdown("### Get Performance Metrics")

    model_select = st.multiselect(
        label='Select trained models for evaluation',
        options=model_options
    )

    if model_select:
        st.write(f"You selected: {model_select}")

        if st.button("Evaluate your selected classification models"):
            for model_name in model_select:
                st.subheader(f"📊 Evaluation: {model_name}")
                model = st.session_state[model_name]
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                st.write(f"**Accuracy:** {acc:.4f}")
                st.write(f"**Precision:** {prec:.4f}")
                st.write(f"**Recall:** {rec:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix - {model_name}")
                st.pyplot(fig)

                # Plot cost history if Logistic Regression
                if model_name == "Logistic Regression":
                    st.markdown("### Training Cost History")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(model.cost_history)
                    ax2.set_title("Training Cost over Epochs")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Cost")
                    st.pyplot(fig2)

    st.write('Continue to Deploy Model')
