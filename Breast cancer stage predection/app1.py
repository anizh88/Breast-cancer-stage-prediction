import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\\Breast cancer stage predection\\ML_MODEL\\gnb_model.pkl'
model = joblib.load(model_path)

# Define expected columns based on the features used in the model
expected_columns = ['Tumor_Size', 'Nodes_Positive', 'Metastasis', 'Histological_Grade', 'ER_Status', 'PR_Status', 'HER2_Status']

def main():
    # Set the title of the web app
    st.title('Breast Cancer Stage Prediction')

    # Add a description
    st.write('Enter data of the patient')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        tumor_size = st.slider('Tumor Size (mm)', 0, 200, 30)
        nodes_positive = st.slider('Number of Positive Nodes', 0, 50, 0)
        metastasis = st.selectbox('Metastasis (0 = No, 1 = Yes)', [0, 1])
        histological_grade = st.selectbox('Histological Grade', [1, 2, 3])
        er_status = st.selectbox('ER Status (0 = Negative, 1 = Positive)', [0, 1])
        pr_status = st.selectbox('PR Status (0 = Negative, 1 = Positive)', [0, 1])
        her2_status = st.selectbox('HER2 Status (0 = Negative, 1 = Positive)', [0, 1])

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Tumor_Size': [tumor_size],
        'Nodes_Positive': [nodes_positive],
        'Metastasis': [metastasis],
        'Histological_Grade': [histological_grade],
        'ER_Status': [er_status],
        'PR_Status': [pr_status],
        'HER2_Status': [her2_status]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            
            # Print raw prediction for debugging
            st.write(f'Raw Prediction: {prediction}')
            
            # Handle case where prediction might be a string
            if len(prediction) > 0:
                prediction_value = prediction[0]
                stage_mapping = {
                    'I': 'Stage I',
                    'II': 'Stage II',
                    'III': 'Stage III',
                    'IV': 'Stage IV'
                }
                
                # Debugging information
                st.write(f'Prediction Value: {prediction_value}')
                
                if prediction_value in stage_mapping:
                    predicted_stage = stage_mapping[prediction_value]
                    st.write(f'Predicted Stage: {predicted_stage}')
                    
                   
                else:
                    st.write('Prediction value is not mapped to a stage.')
            else:
                st.write('No prediction returned.')

if __name__ == '__main__':
    main()
