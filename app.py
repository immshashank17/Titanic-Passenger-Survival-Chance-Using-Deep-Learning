import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")

# Inputs
pclass = st.slider("Enter Passenger Class", 1, 3)

sex = st.selectbox(
    "Enter Passenger Gender",
    ["male", "female"]
)

sibsp = st.slider(
    "Enter Number of Siblings/Spouses",
    0, 8
)

parch = st.slider(
    "Enter Number of Parents/Children",
    0, 6
)

fare = st.number_input(
    "Enter Passenger Fare",
    
)
embarked = st.selectbox(
    "Enter Port of Embarkation for the Journey",
    ["Chebourg", "Queenstown", "Southampton"]
)

# Create DataFrame
data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})
model = load_model("model.h5")
with open("label_encoder.pkl", "rb") as file:
    label = pickle.load(file)
    
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
with open("onehot_encoder.pkl", "rb") as file:
    onehot = pickle.load(file)
    
    
st.write(data["Sex"])
    
data["Sex"] = label.transform(data["Sex"])
embarked = onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked, columns=onehot.get_feature_names_out(['Embarked']))
data=pd.concat([data.drop(columns='Embarked'),embarked],axis=1)

data[["Pclass",  "SibSp", "Parch", "Fare"]] = scaler.transform(data[["Pclass", "SibSp", "Parch", "Fare"]])

y=model.predict(data)
y=y.flatten()[0]
def Chance(y):
    if y> 0.5:
        return "The passenger is likely to survive."
    else:
        return "The passenger is unlikely to survive."

if st.button('Predict Survival Chance'):
    st.write("Survival Probability Chance:",y)
    st.write(Chance(y))

