import numpy as np
import pickle
import streamlit as st

# Loading Saved Model
titanic_survival_model = pickle.load(open('F:/Machine Learning Projects/Titanic Survival Prediction-Deployment/trained_model.sav','rb'))

# Prediction Function
def titanic_prediction(input_data):

    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)

    prediction = titanic_survival_model.predict(input_data)

    if prediction[0] == 1:
        return "✅ Passenger Survived"

    else:
        return "❌ Passenger Did Not Survive"


# ---------------- Streamlit UI ---------------- #
def main():

    st.title(" 🚢 Titanic Survival Prediction")

    Passenger_class = st.text_input("Passenger Class")
    Sex = st.text_input("Sex")
    Age = st.text_input("Age")
    SibSp = st.text_input("Siblings/Spouse")
    Parch = st.text_input("Parents/Children")
    Fare = st.text_input("Fare")
    Embarked = st.text_input("Embarked")

    if st.button("Predict"):

        result = titanic_prediction([
            float(Passenger_class),
            float(Sex),
            float(Age),
            float(SibSp),
            float(Parch),
            float(Fare),
            float(Embarked)
        ])

        st.success(result)

if __name__ == "__main__":
    main()