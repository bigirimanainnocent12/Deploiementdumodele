
import os
import streamlit as st
import joblib
import pandas as pd

# Charger le modèle entraîné
@st.cache_resource
def load_model():
    model_path = "modele.pkl"  # Utilisez un chemin relatif
    if not os.path.exists(model_path):
        st.error(f"Le fichier {model_path} est introuvable. Assurez-vous qu'il est présent dans le répertoire du projet.")
        return None
    try:
        with open(model_path, "rb") as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

# Charger le modèle
model = load_model()

# Vérifiez si le modèle est valide
if model and not hasattr(model, "predict"):
    st.error("Le modèle chargé n'a pas de méthode 'predict'. Vérifiez que vous avez bien enregistré un modèle scikit-learn valide.")
elif not model:
    st.error("Le modèle n'a pas pu être chargé correctement. Veuillez vérifier le fichier.")

# Interface utilisateur
st.title("Déploiement d'un modèle RandomForestRegressor()")

# Ajouter une photo illustrative
image_path = "image.jpg"  # Utilisez un chemin relatif
if os.path.exists(image_path):
    st.image(image_path, caption="Photo illustrative", use_container_width=True)
else:
    st.warning(f"Le fichier {image_path} est introuvable. Assurez-vous qu'il est présent dans le répertoire du projet.")

st.subheader("Simuler vos dépenses médicales")

# Champs d'entrée utilisateur
age = st.number_input("Quel âge avez-vous ?", min_value=0, step=1, format="%d")
sex = st.radio("Quel est votre sexe ?", ["Homme", "Femme"])  # Pas de valeur par défaut
bmi = st.number_input("Quel est votre IMC (Indice de Masse Corporelle) ?", min_value=0.0, step=0.1, format="%.1f")
children = st.number_input("Nombre d'enfants", min_value=0, step=1, format="%d")
smoker = st.radio("Est-ce que vous fumez ?", ["Oui", "Non"])  # Pas de valeur par défaut
region = st.radio("Quelle est votre région ?", ["Nord", "Sud", "Est", "Ouest"])  # Pas de valeur par défaut

if st.button("Envoyer"):
    try:
        # Préparer les données utilisateur
        sex_encoded = True if sex == "Homme" else False  # Homme = 1, Femme = 0
        smoker_encoded = True if smoker == "Oui" else False  # Oui = 1, Non = 0
                 # Création du DataFrame d'entrée
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_encoded],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded],
            'region': [region]
        })

        # Vérification des types et gestion des données manquantes
        if input_data.isnull().values.any():
            st.error("Veuillez remplir tous les champs avant de soumettre.")
        else:
            # Conversion explicite des types pour éviter les erreurs de type (comme 'ufunc isnan')
            input_data = input_data.astype(float)

            # Prédiction avec le modèle
            prediction = model.predict(input_data)
            st.success(f"Les dépenses médicales estimées pour ce client sont : {prediction[0]:.2f} $")
    except Exception as e:
        st.error(f"Une erreur est survenue : {str(e)}")

