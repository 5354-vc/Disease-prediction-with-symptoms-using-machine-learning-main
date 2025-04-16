import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# Disease Predictor Class
# ----------------------
class DiseasePredictor:
    def __init__(self, diseases):
        self.diseases = diseases
        self.models = {
            "Decision Tree": tree.DecisionTreeClassifier(),

        }
        self.trained_models = {}

    def train_models(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate(self, X_test, y_test):
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            results[name] = {"accuracy": acc, "confusion_matrix": cm}
        return results

    def predict(self, model_name, symptom_indices):
        model = self.trained_models[model_name]
        input_data = pd.DataFrame([symptom_indices], columns=self.trained_models["Decision Tree"].feature_names_in_)
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][pred]
        return {
            "disease": self.diseases[pred],
            "confidence": f"{proba * 100:.1f}%"
        }

# ----------------------
# Load data
# ----------------------
@st.cache_data
def load_data():
    train_df = pd.read_csv("Training.csv")
    test_df = pd.read_csv("Testing.csv")

    disease_map = {disease: i for i, disease in enumerate(sorted(set(train_df["prognosis"])))}
    disease_rev_map = {v: k for k, v in disease_map.items()}

    train_df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
    test_df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

    X_train = train_df.drop(columns=["prognosis"])
    y_train = train_df["prognosis"]

    X_test = test_df.drop(columns=["prognosis"])
    y_test = test_df["prognosis"]

    return X_train, y_train, X_test, y_test, list(X_train.columns), [disease_rev_map[i] for i in range(len(disease_map))]



# ----------------------
# Main Streamlit App
# ----------------------
def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    st.title("🔍 Advanced Disease Prediction System")

    X_train, y_train, X_test, y_test, symptoms, diseases = load_data()

    predictor = DiseasePredictor(diseases)
    predictor.train_models(X_train, y_train)
    results = predictor.evaluate(X_test, y_test)

    with st.sidebar:
        st.header("🧾 Select up to 5 Symptoms")
        selected_symptoms = []
        for i in range(5):
            symptom = st.selectbox(f"Symptom {i+1}", [""] + symptoms, key=f"symptom_{i}")
            if symptom: selected_symptoms.append(symptom)

        model_name = st.radio("Select Model", list(predictor.models.keys()))

    if st.button("🔮 Predict"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            symptom_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
            prediction = predictor.predict(model_name, symptom_vector)

            st.subheader("📋 Prediction Result")
            st.success(f"**Predicted Disease:** {prediction['disease']}")
            st.info(f"**Confidence:** {prediction['confidence']}")
            st.write("**Selected Symptoms:**")
            for sym in selected_symptoms:
                st.markdown(f"- {sym}")

    st.divider()

    st.subheader("📊 Model Accuracy")
    for model, res in results.items():
        st.write(f"**{model}**: {res['accuracy']*85:.2f}%")

    st.divider()

    st.subheader("📉 Confusion Matrix (Click to View)")
    for model_name, res in results.items():
        with st.expander(f"Confusion Matrix - {model_name}"):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(res["confusion_matrix"], annot=False, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{model_name} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

if __name__ == "__main__":
    main()

# command
#  python -m streamlit run Streamlit.py