import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title='Titanic Survival AI Predictor', page_icon='🚢', layout='centered'
)


# Load trained pipeline once and cache it
@st.cache_resource
def load_model():
  return joblib.load('titanic_model.joblib')


model = load_model()

# UI Header
st.title('🚢 Titanic Survival AI Predictor')
st.markdown(
    'Enter passenger details below to predict survival probability using the'
    ' trained Random Forest model.'
)
st.divider()

# Input Form
with st.form('passenger_form'):
  col1, col2 = st.columns(2)

  with col1:
    pclass = st.selectbox(
        'Ticket Class (Pclass)',
        options=[1, 2, 3],
        format_func=lambda x: f'{x}st/nd/rd Class',
    )
    sex = st.selectbox('Gender', options=['male', 'female'])
    age = st.slider('Age', min_value=0, max_value=100, value=28)
    fare = st.number_input(
        'Ticket Fare ($)', min_value=0.0, max_value=600.0, value=32.0, step=1.0
    )

  with col2:
    sibsp = st.number_input(
        'Siblings / Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0
    )
    parch = st.number_input(
        'Parents / Children Aboard (Parch)', min_value=0, max_value=10, value=0
    )
    embarked = st.selectbox(
        'Port of Embarkation',
        options=['S', 'C', 'Q'],
        format_func=lambda x: {
            'S': 'Southampton (S)',
            'C': 'Cherbourg (C)',
            'Q': 'Queenstown (Q)',
        }[x],
    )

  submitted = st.form_submit_button('Predict Survival', use_container_width=True)

# Inference & Output Display
if submitted:
  input_data = pd.DataFrame([{
      'Pclass': pclass,
      'Sex': sex,
      'Age': age,
      'SibSp': sibsp,
      'Parch': parch,
      'Fare': fare,
      'Embarked': embarked,
  }])

  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  st.divider()
  if prediction == 1:
    st.success(
        f'### ✅ Prediction: Survived (Survival Probability:'
        f' {probability:.1%})'
    )
  else:
    st.error(
        '### ❌ Prediction: Did Not Survive (Survival Probability:'
        f' {probability:.1%})'
    )
