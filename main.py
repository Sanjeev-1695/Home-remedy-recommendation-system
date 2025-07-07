import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os

os.environ["GROQ_API_KEY"] = "API-KEY" #  Replace ur api key here 

chat = ChatGroq(model="llama3-70b-8192")
df = pd.read_csv("dataset.csv")
ALLOWED_DISEASES = [
    "Common Cold", "Influenza (Flu)", "Pneumonia", "Gastroenteritis", "Irritable Bowel Syndrome (IBS)",
    "Diabetes Mellitus", "UTI", "Hypertension", "Migraine", "Hypothyroidism", "Hyperthyroidism",
    "Anemia", "Heart Failure", "Asthma", "Allergic Rhinitis", "Sinusitis", "Appendicitis", "Peptic Ulcer",
    "Tuberculosis", "Liver Disease (Hepatitis)", "Kidney Disease", "Depression", "Anxiety Disorders",
    "Eczema/Dermatitis", "Lupus", "Rheumatoid Arthritis", "Chikungunya", "Dengue", "Malaria", "Sleep Apnea",
    "Sarcoidosis", "Carcinoid Syndrome", "Fibromyalgia", "Guillain-Barré Syndrome", "Multiple Sclerosis",
    "Hashimoto's Thyroiditis", "Chronic Fatigue Syndrome", "Lyme Disease", "Rocky Mountain Spotted Fever",
    "Chronic Hepatitis", "Cytomegalovirus Infection", "HIV/AIDS", "Mononucleosis", "Chronic Kidney Disease",
    "Systemic Lupus Erythematosus (Lupus)", "Vasculitis", "Myasthenia Gravis", "Pernicious Anemia",
    "Sjogren’s Syndrome", "Psoriasis", "PCOS", "Psoriatic Arthritis", "GERD", "Sciatica", "Gout",
    "Hepatitis B", "Shingles", "Bronchitis", "Insomnia", "Rheumatic Fever",
    "Chronic Obstructive Pulmonary Disease (COPD)", "Peptic Stricture", "Bell’s Palsy",
    "Tinnitus", "Meniere’s Disease", "Multiple Myeloma", "Celiac Disease", "Sickle Cell Disease",
    "Alzheimer’s Disease"
]
SYMPTOM_OPTIONS = [
    "Fever", "Fatigue", "Weight loss", "Loss of appetite", "Cough (dry or wet)",
    "Shortness of breath", "Sore throat", "Runny nose", "Blocked nose", "Nausea",
    "Vomiting", "Diarrhea", "Constipation", "Stomach pain", "Chest pain", "Palpitations",
    "Swelling in legs/feet", "Headache", "Dizziness", "Numbness", "Tingling", "Itching",
    "Skin rashes", "Joint pain", "Frequent urination", "Muscle pain", "Chills",
    "Night sweats", "Blurred vision", "Difficulty sleeping", "Sneezing",
    "Loss of taste or smell", "Cramps", "Dry mouth"
]
def get_disease_name(age, pre_conditions, symptoms):
    allowed_list = ", ".join(ALLOWED_DISEASES)
    messages = [
        SystemMessage(content=f"""You are a smart medical assistant. 
Only respond with one of the following disease names: {allowed_list}. 
Based on the user's age, symptoms, and pre-existing conditions, predict the most likely disease name.
Do NOT include any explanation or extra words — just the disease name exactly as in the list."""),
        HumanMessage(content=f"Age: {age}\nPre-conditions: {pre_conditions}\nSymptoms: {symptoms}\nWhat is the most likely disease?")
    ]
    response = chat.invoke(messages)
    predicted = response.content.strip().lower()

    allowed_map = {d.lower(): d for d in ALLOWED_DISEASES}
    if predicted in allowed_map:
        return allowed_map[predicted]
    else:
        return None


# ---------- Remedy Selector ----------
def get_remedy(disease, age, allergy, diet_pref):
    df_clean = df.copy()
    df_clean["Allergies"] = df_clean["Allergies"].fillna("").str.lower()
    df_clean["Dietary Preferences"] = df_clean["Dietary Preferences"].fillna("").str.lower()
    df_clean["Suitable Age Group"] = df_clean["Suitable Age Group"].fillna("0+")

    # -- Diet Preference Fix --
    if diet_pref.lower() == "vegetarian":
        diet_mask = df_clean["Dietary Preferences"].isin(["vegetarian", "vegan"])
    else:
        diet_mask = df_clean["Dietary Preferences"] == diet_pref.lower()
        allergy = allergy.strip().lower()
    if allergy in ["none", "no allergies", "no allergy", "no"]:
        allergy_mask = pd.Series([True] * len(df_clean))
    else:
        allergy_mask = ~df_clean["Allergies"].str.contains(allergy)
    def is_age_suitable(age_group):
        try:
            digits = ''.join(filter(str.isdigit, str(age_group)))
            min_age = int(digits)
            return age >= min_age
        except:
            return False

    age_mask = df_clean["Suitable Age Group"].apply(is_age_suitable)
    final_df = df_clean[
        (df_clean["Disease Name"].str.lower() == disease.lower()) &
        diet_mask &
        allergy_mask &
        age_mask
    ]

    if final_df.empty:
        st.warning("⚠️ No suitable remedy found for your preferences. Try adjusting allergy, age, or diet filters.")
        return None, None
    else:
        remedy = final_df.sample(1).iloc[0]
        return remedy["Remedy Name"], remedy

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Symptom Checker & Remedy Suggestion", layout="centered")
st.title("🧠 Symptom Checker & 🪴 Home Remedy Recommender")

st.subheader("👤 Your Info")
age = st.number_input("Your Age", min_value=1, max_value=120, step=1)
pre_conditions = st.text_input("Pre-existing Conditions (optional)")
allergy = st.text_input("Any Allergies? (type 'none' if none)", placeholder="e.g. lactose, pollen")
diet_pref = st.selectbox("Dietary Preference", ["vegetarian", "non-vegetarian", "vegan"])

st.subheader("📝 Select Your Symptoms")
selected_symptoms = st.multiselect("Choose from common symptoms", SYMPTOM_OPTIONS)

if st.button("Predict Disease & Remedy"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    elif allergy.strip() == "":
        st.warning("Please enter allergy info (type 'none' if no allergies).")
    else:
        symptoms_text = ", ".join(selected_symptoms)
        with st.spinner("Analyzing..."):
            disease_result = get_disease_name(age, pre_conditions, symptoms_text)

        if disease_result is None:
            st.error(" The predicted disease is not supported yet. Try different symptoms.")
        else:
            st.success(f"🩺 Most Likely Disease: **{disease_result}**")
            remedy_name, remedy_info = get_remedy(disease_result, age, allergy, diet_pref)

            if remedy_info is not None:
                st.markdown(f"### 🌿 Suggested Home Remedy: **{remedy_name}**")
                st.markdown(f"""
                **Ingredients Used:** {remedy_info['Ingredients']}  
                **Preparation Method:** {remedy_info['Preparation Method']}  
                **Side Effects:** {remedy_info['Side Effects']}  
                **Suitable Age Group:** {remedy_info['Suitable Age Group']}  
                """)
            else:
                st.warning("No suitable remedy found for your preferences. Try changing allergy or diet filter.")
