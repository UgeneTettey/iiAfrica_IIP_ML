import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


# load the model
# model = pickle.load(open('new_xgb_model.pkl', 'rb'))
import os
model_path = os.path.join(os.getcwd(), 'C:/Users/Eugene/Desktop/iiAfrica/IIP/ML/day_4/model_deploymentxgb_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# categorical features
categorical_features = {
    'region':['East Africa','West Africa','Central Africa'],
    'gender':['Male', 'Female'],
    'credit_card':['yes', 'no'],
    'active_member':['yes', 'no']
}

# define ecndoder dictionary
encoder_dict = {feature: LabelEncoder().fit(values) for feature, values in categorical_features.items()}

# ==================================================================
# Streamlit app configuration
st.set_page_config(page_title="Customer Eligibility Prediction", page_icon=":mag:", layout="centered", initial_sidebar_state="expanded")



# title and header
st.title('Loan Eligibility Prediction')
st.sidebar.header('Customer Details')

input_data = {}


input_data['credit_score'] = st.sidebar.number_input('Credit Score', min_value=0, step=1, format="%d")
input_data['region'] = st.sidebar.selectbox('Region', options=categorical_features['region'])
input_data['gender'] = st.sidebar.selectbox('Gender', options=categorical_features['gender'])
input_data['age'] = st.sidebar.number_input('Age', min_value=0, max_value=120, step=1, format="%d")
input_data['tenure'] = st.sidebar.number_input('Tenure', min_value=0, max_value=100, step=1, format="%d")
input_data['balance'] = st.sidebar.number_input('Account Balance')
input_data['products_number'] = st.sidebar.number_input('Number of Products', min_value=0, max_value=20, step=1, format="%d")
input_data['credit_card'] = st.sidebar.selectbox('Credit Card', options=categorical_features['credit_card'])
input_data['active_member'] = st.sidebar.selectbox('Active Member', options=categorical_features['active_member'])
input_data['estimated_salary'] = st.sidebar.number_input('Estimated Salary')




# convert input data into dataframe
input_df = pd.DataFrame([input_data])



# Encode categorical feature
for feature, encoder in encoder_dict.items():
    input_df[feature] = encoder.transform(input_df[feature])




# Creating the Predict button
if st.button('PREDICT'):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            # st.write('The customer is likely to churn.')
            st.markdown("<h3 style='color: red;'>The customer is not eligible for loan.</h3>", unsafe_allow_html=True)
        else:
            # st.write('The customer is not likely to churn.')
            st.markdown("<h3 style='color: green;'>The customer is eligible for loan.</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.write("Error during prediction:", str(e))


# Additional sections
st.markdown("## How It Works")
st.write("""
The Bank Customer Eligibilty Prediction app allows you to input customer details and predict the eligibility of clients for loans.
Simply enter the required details in the sidebar and click the 'PREDICT' button.
""")