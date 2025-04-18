import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "model.pkl"

# Train and save the model
def train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(clf, MODEL_PATH)
    return clf, iris.target_names, acc

# Load model
@st.cache_resource
def load_model():
    try:
        clf = joblib.load(MODEL_PATH)
        iris = load_iris()
        return clf, iris.target_names
    except:
        return train_model()[:2]

# UI Starts Here
st.title("🌸 Iris Species Classifier")
st.write("Predict flower species or retrain the model below.")

# Train or Load model
retrain = st.button("🔁 Retrain Model")
if retrain:
    model, target_names, accuracy = train_model()
    st.success(f"✅ Model retrained! Accuracy: {accuracy:.2f}")
else:
    model, target_names = load_model()

st.subheader("📥 Input Flower Measurements")

# Inputs
sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.1, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.5, step=0.1)
petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 1.4, step=0.1)
petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2, step=0.1)

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
    sepal_length, sepal_width, petal_length, petal_width
]], columns=[
    'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
])
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]
    st.success(f"🌼 Predicted Species: **{target_names[prediction].capitalize()}**")
    st.write("🔬 Prediction Confidence:")
    for i, name in enumerate(target_names):
        st.write(f"- {name.capitalize()}: {probas[i]:.2%}")
