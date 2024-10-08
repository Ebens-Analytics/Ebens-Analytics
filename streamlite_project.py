import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or use pickle



# Load the saved model
model = joblib.load('logistic_model.joblib')

def instagram_acct_pred(input_data):
    numpy_array = np.asarray(input_data)
    reshaped_array = numpy_array.reshape(1, -1)
    
    prediction = model.predict(reshaped_array)
    if prediction == 0:
        return 'Account is not fake'
    else:
        return 'Account might be fake'


def main():
    st.title('Instagram account Prediction Model :')
    
    profile_pic = st.radio("Profile Picture:", (1, 0))
    nums_length_username = st.slider('nums_length_username :',0, 100)
    fullname_words = st.slider('fullname_words :', 0, 12)
    nums_length_fullname = st.number_input('nums/length_fullname :', min_value=0, max_value=100, value=0)
    name_username = st.radio('name is same as username :', (1, 0))
    description_length = st.slider('description_length :', 0, 150)
    external_URL = st.radio('external_URL :', (1, 0))
    private = st.radio("private account:", (1, 0))
    posts = st.slider('posts :', 0, 7389)
    followers = st.slider("how many followers :", 0, 15338538)
    follows = st.slider('Follows how many people :', 0, 7500)

    Account = ''

    if st.button('Predict'):
        account_prediction = instagram_acct_pred(
            [int(profile_pic), float(nums_length_username), int(fullname_words), float(nums_length_fullname), int(name_username), int(description_length), int(external_URL), int(private), int(posts), int(followers), int(follows)]
            )
    st.success(account_prediction)
    
if __name__ == '__main__':
    main()
