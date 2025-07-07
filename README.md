# 🌿 Ayurvedic Home Remedy Recommendation System

A smart and interactive health assistant built with **Streamlit**, **LLaMA 3 (via Groq API)**, and a **custom Ayurvedic remedies dataset**. This application predicts the most likely disease based on user inputs (age, symptoms, pre-existing conditions) and then suggests safe, suitable, and natural home remedies from Ayurveda.

---

## 🚀 Features

- 🧠 **Disease Prediction** using Groq's LLaMA 3 model.
- 🌱 **Ayurvedic Remedy Suggestions** tailored to age, allergies, and dietary preferences.
- 📋 Built-in symptom selection with 30+ common symptoms.
- ⚖️ Rule-based filtering to match remedies based on user profile.
- 🖥️ Easy-to-use Streamlit web interface.

---

## 🧰 Technologies Used

- **Frontend/UI**: Streamlit  
- **Backend Logic**: Python, Pandas  
- **AI/NLP**: Langchain + Groq API (LLaMA 3)  
- **Data**: Custom-built `dataset.csv` of 400+ Ayurvedic remedies  
- **Deployment**: Local / Streamlit Cloud compatible

---

## 📊 Dataset Columns

| Column Name             | Description                                      |
|-------------------------|--------------------------------------------------|
| Disease Name            | Name of the condition being treated              |
| Remedy Name             | Name/title of the Ayurvedic remedy              |
| Ingredients             | Natural ingredients used                        |
| Preparation Method      | Step-by-step preparation instructions           |
| Side Effects            | Possible side effects                           |
| Suitable Age Group      | Recommended minimum age                         |
| Remedy Category         | Tea, Paste, Oil, etc.                           |
| Time to Prepare         | Time required for preparation                   |
| Severity Level          | Mild / Moderate / Severe                        |
| Allergies               | Ingredients that may trigger allergies          |
| Dietary Preferences     | Vegan, Vegetarian, Non-Vegetarian               |

---

## 🖼️ UI Preview

> 🧠 Symptom Checker  
> 🪴 Home Remedy Recommender  
> ⚙️ Custom filters: Age, Allergies, Diet  

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the Repo

git clone https://github.com/Sanjeev-1695/Home-remedy-recommendation-system.git
cd Home-remedy-recommendation-system

### 3. Install Requirements
pip install -r requirements.txt

### 4. Add Your Groq API Key
Replace the placeholder in the script with your actual Groq API Key:

os.environ["GROQ_API_KEY"] = "your_actual_groq_api_key"

### 5. Run the App
streamlit run app.py
