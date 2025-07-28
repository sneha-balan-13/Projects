import sys
import os

# Add the parent directory of app.py (which is the project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from src.model import load_model
from src.preprocessing import preprocess_text

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.pkl')
model = load_model(model_path)

# SQLite setup
conn = sqlite3.connect('predictions.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        timestamp TEXT,
        subject TEXT,
        description TEXT,
        priority TEXT,
        confidence REAL
    )
''')
conn.commit()

# Priority mapping
priority_map = {0: 'Low', 1: 'Medium', 2: 'High'}

# Predict function
def predict_priority(subject, description):
    combined_text = preprocess_text(subject, description)
    prediction = model.predict([combined_text])[0]
    confidence = max(model.predict_proba([combined_text])[0])
    priority_label = priority_map[prediction]
    return priority_label, confidence

# Log prediction
def log_prediction(subject, description, priority_label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", 
              (timestamp, subject, description, priority_label, round(confidence, 2)))
    conn.commit()

# App title
st.title("🎫 Customer Support Ticket Prioritization")
st.caption("Powered by Machine Learning | Logs saved to SQLite")

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📄 Batch Prediction"])

with tab1:
    st.subheader("Single Ticket Prediction")
    st.write("Enter the **subject** and **description** of a customer support ticket:")

    subject = st.text_input("Subject", key="single_subject")
    description = st.text_area("Description", key="single_description")

    if st.button("Predict Priority", key="single_predict"):
        if subject and description:
            priority_label, confidence = predict_priority(subject, description)
            st.success(f"Predicted Priority: **{priority_label}** (Confidence: {confidence:.2f})")
            log_prediction(subject, description, priority_label, confidence)
        else:
            st.warning("Please fill in both fields.")

with tab2:
    st.subheader("Batch Prediction via CSV Upload")
    st.write("Upload a CSV file with **Subject** and **Description** columns.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv", key="batch_upload")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'Subject' not in df.columns or 'Description' not in df.columns:
            st.error("CSV must contain 'Subject' and 'Description' columns.")
        else:
            results = []
            for _, row in df.iterrows():
                subj, desc = str(row['Subject']), str(row['Description'])
                label, conf = predict_priority(subj, desc)
                log_prediction(subj, desc, label, conf)
                results.append([subj, desc, label, round(conf, 2)])

            result_df = pd.DataFrame(results, columns=['Subject', 'Description', 'Predicted Priority', 'Confidence'])
            st.write("📊 Prediction Results:")
            st.dataframe(result_df)

st.markdown("---")

# Prediction history
st.subheader("📜 Prediction History")
if st.button("View Prediction Log"):
    logs = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    st.dataframe(logs)
