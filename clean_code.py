import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Disease Prediction System")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f8ff")
        
        # Initialize data
        self.symptoms = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever',
                        'yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
                        'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision',
                        'phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose',
                        'congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
                        'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
                        'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity',
                        'swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid',
                        'brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts',
                        'drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
                        'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
                        'spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
                        'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine',
                        'passage_of_gases','internal_itching','toxic_look_(typhos)','depression',
                        'irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
                        'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite',
                        'polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
                        'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
                        'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
                        'blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking',
                        'pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting',
                        'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
                        'yellow_crust_ooze']
        
        self.diseases = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
                        'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma',
                        'Hypertension','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
                        'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
                        'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis',
                        'Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
                        'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia',
                        'Osteoarthristis','Arthritis','(vertigo) Paroymsal Positional Vertigo','Acne',
                        'Urinary tract infection','Psoriasis','Impetigo']
        
        # Load and prepare data
        self.load_data()
        
        # Initialize predictor
        self.predictor = DiseasePredictor(self.diseases)
        self.predictor.train_models(self.X_train, self.y_train)
        self.results = self.predictor.evaluate(self.X_test, self.y_test)
        
        # Build GUI
        self.create_widgets()
    
    def load_data(self):
        try:
            # Load training data
            train_df = pd.read_csv("Training.csv")
            train_df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
            self.X_train = train_df[self.symptoms]
            self.y_train = train_df["prognosis"].astype(int).values
            
            # Load testing data
            test_df = pd.read_csv("Testing.csv")
            test_df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
            self.X_test = test_df[self.symptoms]
            self.y_test = test_df["prognosis"].astype(int).values
             
                # Calculate entropy
            self.entropy_value = entropy(pd.Series(self.y_train).value_counts(normalize=True), base=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        # Header
        header = tk.Label(self.root, text="Disease Prediction System", 
                         font=("Helvetica", 24, "bold"), bg="#f0f8ff", fg="#2c3e50")
        header.pack(pady=20)
        
        # Entropy display
        entropy_frame = tk.Frame(self.root, bg="#f0f8ff")
        entropy_frame.pack(pady=10)
        tk.Label(entropy_frame, text=f"Dataset Entropy: {self.entropy_value:.4f}", 
                font=("Helvetica", 12), bg="#f0f8ff").pack()
        # Symptoms selection
        symptoms_frame = tk.LabelFrame(self.root, text="Select Symptoms", 
                                      font=("Helvetica", 14), bg="#f0f8ff")
        symptoms_frame.pack(pady=10, padx=20, fill="x")
        
        self.symptom_vars = []
        for i in range(5):
            frame = tk.Frame(symptoms_frame, bg="#f0f8ff")
            frame.pack(fill="x", pady=5)
            tk.Label(frame, text=f"Symptom {i+1}:", width=12, anchor="w", 
                    bg="#f0f8ff").pack(side="left")
            var = tk.StringVar()
            combobox = ttk.Combobox(frame, textvariable=var, values=sorted(self.symptoms), 
                                   state="readonly", width=40)
            combobox.pack(side="left", padx=5)
            self.symptom_vars.append(var)
        
        # Prediction buttons
        button_frame = tk.Frame(self.root, bg="#f0f8ff")
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Decision Tree", command=lambda: self.predict("Decision Tree"), 
                 bg="#3498db", fg="white", font=("Helvetica", 12)).pack(side="left", padx=10)
        tk.Button(button_frame, text="Random Forest", command=lambda: self.predict("Random Forest"), 
                 bg="#2ecc71", fg="white", font=("Helvetica", 12)).pack(side="left", padx=10)
        tk.Button(button_frame, text="Naive Bayes", command=lambda: self.predict("Naive Bayes"), 
                 bg="#e74c3c", fg="white", font=("Helvetica", 12)).pack(side="left", padx=10)
        
        # Results display
        results_frame = tk.LabelFrame(self.root, text="Prediction Results", 
                                     font=("Helvetica", 14), bg="#f0f8ff")
        results_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Prediction tab
        self.prediction_tab = tk.Frame(self.notebook, bg="#f0f8ff")
        self.notebook.add(self.prediction_tab, text="Prediction")
        
        self.result_text = tk.Text(self.prediction_tab, height=10, width=80, 
                                  font=("Helvetica", 12), wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model info tab
        self.model_info_tab = tk.Frame(self.notebook, bg="#f0f8ff")
        self.notebook.add(self.model_info_tab, text="Model Performance")
        
        self.model_info_text = tk.Text(self.model_info_tab, height=10, width=80,
                                      font=("Helvetica", 12), wrap="word")
        self.model_info_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Show model accuracies
        self.update_model_info()
        
        # Confusion matrix button
        tk.Button(self.root, text="Show Confusion Matrices", command=self.show_confusion_matrices,
                 bg="#9b59b6", fg="white", font=("Helvetica", 12)).pack(pady=10)
    
    def predict(self, model_name):
        selected_symptoms = [var.get() for var in self.symptom_vars if var.get()]
        
        if not selected_symptoms:
            messagebox.showwarning("Warning", "Please select at least one symptom!")
            return
        
        try:
            # Convert symptoms to binary format
            symptom_vector = [1 if symptom in selected_symptoms else 0 for symptom in self.symptoms]
            
            # Get prediction
            prediction = self.predictor.predict(model_name, symptom_vector)
            accuracy = self.results[model_name]["accuracy"]
            
            # Update prediction tab
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Model: {model_name}\n", "title")
            self.result_text.insert(tk.END, f"Accuracy: {accuracy*100:.2f}%\n")
            self.result_text.insert(tk.END, f"Predicted Disease: {prediction['disease']}\n")
            self.result_text.insert(tk.END, f"Confidence: {prediction['confidence']}\n\n")
            self.result_text.insert(tk.END, "Selected Symptoms:\n", "subtitle")
            for symptom in selected_symptoms:
                self.result_text.insert(tk.END, f"- {symptom}\n")
            
            # Configure text tags
            self.result_text.tag_config("title", font=("Helvetica", 14, "bold"))
            self.result_text.tag_config("subtitle", font=("Helvetica", 12, "bold"))
            
            # Switch to prediction tab
            self.notebook.select(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def update_model_info(self):
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, "Model Performance Metrics:\n\n", "title")
        
        for name, result in self.results.items():
            self.model_info_text.insert(tk.END, f"{name}:\n", "subtitle")
            self.model_info_text.insert(tk.END, f"Accuracy: {result['accuracy']*100:.2f}%\n\n")
        
        self.model_info_text.tag_config("title", font=("Helvetica", 14, "bold"))
        self.model_info_text.tag_config("subtitle", font=("Helvetica", 12, "bold"))
    
    def show_confusion_matrices(self):
        for model_name, result in self.results.items():
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix - {model_name}\nAccuracy: {result['accuracy']*100:.2f}%")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
            # Show in new window
            top = tk.Toplevel(self.root)
            top.title(f"Confusion Matrix - {model_name}")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            plt.close(fig)

class DiseasePredictor:
    def __init__(self, diseases):
        self.diseases = diseases
        self.models = {
            "Decision Tree": tree.DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB()
        }
        self.trained_models = {}
    
    def train_models(self, X_train, y_train):
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
    
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
            "confidence": f"{proba*100:.1f}%"
        }

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()