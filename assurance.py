
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os

# Supprimer les avertissements de version sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Coûts d'Assurance Maladie",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un style professionnel
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stAlert {
        margin-top: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Sidebar header */
    .css-1d391kg .css-1v0mbdj {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Section headers in sidebar */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #e0e6ed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar-section h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* BMI display */
    .bmi-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .bmi-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-normal { background: #d4edda; color: #155724; padding: 0.5rem; border-radius: 8px; }
    .status-warning { background: #fff3cd; color: #856404; padding: 0.5rem; border-radius: 8px; }
    .status-danger { background: #f8d7da; color: #721c24; padding: 0.5rem; border-radius: 8px; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stSlider > div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e6ed;
    }
    
    .stCheckbox > label {
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e0e6ed;
        margin: 0.2rem 0;
    }
    
    .stSelectbox > div {
        background: white;
        border-radius: 8px;
    }
    
    .stNumberInput > div {
        background: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle (basée sur application.py)
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

# Fonction de simulation si le modèle n'est pas disponible
def simulate_prediction(age, sex, bmi, children, smoker, region):
    """Simulation de prédiction basée sur les patterns du modèle"""
    base_cost = 3000
    
    # Facteur âge
    base_cost += age * 50
    
    # Facteur IMC
    if bmi > 30:
        base_cost += (bmi - 30) * 200
    elif bmi < 18.5:
        base_cost += (18.5 - bmi) * 100
    
    # Facteur fumeur (impact majeur)
    if smoker:
        base_cost *= 2.5
    
    # Facteur enfants
    base_cost += children * 500
    
    # Facteur sexe
    if sex:
        base_cost *= 1.1
    
    # Facteur région
    region_multiplier = {
        'Nord': 1.0,
        'Sud': 0.95,
        'Est': 1.05,
        'Ouest': 1.02
    }
    base_cost *= region_multiplier.get(region, 1.0)
    
    return max(1000, round(base_cost))



# Interface principale
def main():
    # En-tête
    st.markdown("""
    # 🏥 Prédiction des Coûts d'Assurance Maladie

    """)
    
    # Informations sur le modèle
    with st.expander("ℹ️ À propos de cette prédiction", expanded=False):
        st.markdown("""
        **Notre modèle utilise un algorithme de Random Forest** entraîné sur plus de 27 000 données d'assurance 
        pour prédire vos coûts médicaux annuels.
        
        **Facteurs pris en compte :**
        - 👤 Âge et sexe
        - 📊 Indice de Masse Corporelle (IMC)
        - 🚬 Statut tabagique
        - 👶 Nombre d'enfants à charge
        - 🌍 Région géographique
        
        **Précision du modèle :** R² > 0.85
        """)
    
    # Sidebar pour les inputs
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.5rem;">📋 Vos Informations</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Remplissez tous les champs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Section Informations Personnelles
        st.markdown("""
        <div class="sidebar-section">
            <h3>👤 Informations Personnelles</h3>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.number_input(
            "**Quel âge avez-vous ?**",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            format="%d",
            help="Votre âge actuel"
        )
        
        # Modification : utilisation de radio buttons comme dans application.py
        sex = st.radio(
            "**Quel est votre sexe ?**",
            ["Homme", "Femme"],
            index=0,  # Homme par défaut
            help="Sélectionnez votre sexe"
        )
        
        # Section Informations de Santé
        st.markdown("""
        <div class="sidebar-section">
            <h3>🏥 Informations de Santé</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input(
                "**Taille (cm)**",
                min_value=140,
                max_value=220,
                value=170,
                step=1
            )
        
        with col2:
            weight = st.number_input(
                "**Poids (kg)**",
                min_value=40,
                max_value=200,
                value=70,
                step=1
            )
        
        # Calcul et affichage de l'IMC
        bmi = weight / ((height/100) ** 2)
        
        st.markdown(f"""
        <div class="bmi-display">
            <div style="font-size: 0.9rem; opacity: 0.9;">IMC Calculé</div>
            <div class="bmi-value">{bmi:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interprétation de l'IMC avec style
        if bmi < 18.5:
            st.markdown('<div class="status-warning">⚖️ <strong>Poids insuffisant</strong></div>', unsafe_allow_html=True)
        elif 18.5 <= bmi < 25:
            st.markdown('<div class="status-normal">✅ <strong>Poids normal</strong></div>', unsafe_allow_html=True)
        elif 25 <= bmi < 30:
            st.markdown('<div class="status-warning">⚠️ <strong>Surpoids</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-danger">🚨 <strong>Obésité</strong></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Modification : utilisation de radio buttons comme dans application.py
        smoker = st.radio(
            "**Est-ce que vous fumez ?**",
            ["Non", "Oui"],
            index=0,  # Non par défaut
            help="Cochez si vous fumez régulièrement"
        )
        
        # Section Informations Familiales
        st.markdown("""
        <div class="sidebar-section">
            <h3>👨‍👩‍👧‍👦 Informations Familiales</h3>
        </div>
        """, unsafe_allow_html=True)
        
        children = st.number_input(
            "**Nombre d'enfants**",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            format="%d",
            help="Nombre d'enfants de moins de 18 ans"
        )
        
        # Section Localisation
        st.markdown("""
        <div class="sidebar-section">
            <h3>🌍 Localisation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Modification : utilisation de radio buttons comme dans application.py
        region = st.radio(
            "**Quelle est votre région ?**",
            ["Nord", "Sud", "Est", "Ouest"],
            index=0,  # Nord par défaut
            help="Votre région de résidence principale"
        )
        
        # Espacement avant le bouton
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Bouton de prédiction stylé
        predict_button = st.button(
            "🔮 CALCULER MA PRÉDICTION",
            type="primary",
            use_container_width=True,
            help="Cliquez pour obtenir votre estimation de coûts d'assurance"
        )
    
    # Zone principale - Résultats
    if predict_button:
        try:
            # Afficher les données saisies par le client (comme dans application.py)
            st.subheader("📊 Données saisies par le client :")
            donne = pd.DataFrame({
                "Age": [age],
                "Sexe": [sex],
                "Indice de Masse Corporelle": [f"{bmi:.1f}"],
                "Nombre d'enfants": [children],
                "Fumeur": [smoker],
                "Région": [region]
            })
            st.write(donne)
            
            # Chargement du modèle
            model = load_model()
            
            # Préparation des données avec encodage (basé sur application.py)
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
            
            # Prédiction
            if model and hasattr(model, "predict"):
                prediction = model.predict(input_data)[0]
                st.success("✅ Prédiction réalisée avec le modèle IA")
                
                # Affichage du résultat principal
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>💰 Coût Annuel Estimé</h2>
                        <h1 style="font-size: 3em; margin: 1rem 0;">
                            {prediction:,.2f} $
                        </h1>
                        <p>Estimation basée sur vos informations personnelles</p>
                    </div>
                    """, unsafe_allow_html=True)
                

                
            else:
                # Utilisation de la simulation
                st.warning("⚠️ Modèle non disponible. Utilisation du mode simulation.")
                prediction = simulate_prediction(age, sex_encoded, bmi, children, smoker_encoded, region)
                
                # Affichage du résultat principal
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>💰 Coût Annuel Estimé (Simulation)</h2>
                        <h1 style="font-size: 3em; margin: 1rem 0;">
                            {prediction:,.0f} $
                        </h1>
                        <p>Estimation basée sur vos informations personnelles</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")
        
        # Disclaimer
        st.markdown("""
        ---
        **⚠️ Avertissement :** Cette prédiction est fournie à titre informatif uniquement. 
        Les coûts réels peuvent varier selon de nombreux facteurs non pris en compte par ce modèle. 
        Consultez un professionnel de l'assurance pour une évaluation précise.
        """)
    
    else:
        # Message d'instruction simple
        st.info("👈 Remplissez vos informations dans la barre latérale et cliquez sur 'Calculer ma prédiction' pour obtenir votre estimation.")

if __name__ == "__main__":
    main()

