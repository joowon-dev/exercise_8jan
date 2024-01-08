import numpy as np
import pickle
import pandas as pd
import streamlit as st 

# Load your models (replace 'reg_RF.pkl' and 'other_model.pkl' with your actual model file names)
pickle_in_rf = open("reg_RF.pkl", "rb")
rf_model = pickle.load(pickle_in_rf)

pickle_in_other = open("reg_LGBM.pkl", "rb")
other_model = pickle.load(pickle_in_other)

def predict_std_perf(model, hour_s, pre_score):
    prediction = model.predict([[hour_s, pre_score]])
    print(prediction)
    return prediction

def main():
    st.title("Student Performance Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Student Performance Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    hour_s = st.text_input("Hour of Study", "Type Here")
    pre_score = st.text_input("Previous Score", "Type Here")

    # Model selection checkboxes
    use_rf_model = st.checkbox('Use Random Forest Model', value=True)
    use_other_model = st.checkbox('Use Other Model')

    result = ""
    if st.button("Predict"):
        if use_rf_model:
            result = predict_std_perf(rf_model, hour_s, pre_score)
        elif use_other_model:
            result = predict_std_perf(other_model, hour_s, pre_score)

    st.success('The output is {}'.format(result))

    if st.button("About"):
        st.text("Lets Learn Python")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
