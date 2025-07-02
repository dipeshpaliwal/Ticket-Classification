import pandas as pd
import spacy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Streamlit config
st.set_page_config(page_title="Ticket Classifier", layout="wide")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_excel("tickets.xls", engine="xlrd")

# Drop missing values
df = df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'])

# Extract entities
@st.cache_data
def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

df['entities'] = df['ticket_text'].apply(extract_entities)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['ticket_text'])

# Targets
y_issue = df['issue_type']
y_urgency = df['urgency_level']

# Split data
X_train, X_test, y_train_issue, y_test_issue = train_test_split(X, y_issue, test_size=0.2, random_state=42)
_, _, y_train_urgency, y_test_urgency = train_test_split(X, y_urgency, test_size=0.2, random_state=42)

# Train classifiers
clf_issue = RandomForestClassifier()
clf_issue.fit(X_train, y_train_issue)

clf_urgency = RandomForestClassifier()
clf_urgency.fit(X_train, y_train_urgency)

# ---- Streamlit UI ----

st.title("🎫 Ticket Classification & Entity Extraction App")
st.markdown("Enter a support ticket below. The model will predict **issue type**, **urgency level**, and extract named entities.")

user_input = st.text_area("Enter ticket text", height=150)

if st.button("Classify Ticket"):
    if user_input.strip():
        vec = tfidf.transform([user_input])
        issue = clf_issue.predict(vec)[0]
        urgency = clf_urgency.predict(vec)[0]
        ents = extract_entities(user_input)

        st.subheader("🔍 Predicted Issue Type")
        st.success(issue)

        st.subheader("🚦 Predicted Urgency Level")
        st.success(urgency)

        st.subheader("🧠 Extracted Entities")
        st.json(ents)
    else:
        st.warning("Please enter a ticket description.")

# ---- Sidebar options ----

st.sidebar.title("📊 Explore Model & Data")

# Distribution plots
if st.sidebar.checkbox("Show Ticket Distributions"):
    st.subheader("Ticket Distribution by Issue Type")
    fig1 = plt.figure(figsize=(6, 3))
    df['issue_type'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Issue Type Distribution")
    plt.xlabel("Issue Type")
    plt.ylabel("Count")
    st.pyplot(fig1)

    st.subheader("Ticket Distribution by Urgency Level")
    fig2 = plt.figure(figsize=(6, 3))
    df['urgency_level'].value_counts().plot(kind='bar', color='salmon')
    plt.title("Urgency Level Distribution")
    plt.xlabel("Urgency Level")
    plt.ylabel("Count")
    st.pyplot(fig2)

# Feature importances
if st.sidebar.checkbox("Show Feature Importances (Issue Type)"):
    st.subheader("Top TF-IDF Feature Importances")
    importances = clf_issue.feature_importances_
    feature_names = tfidf.get_feature_names_out()
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    top_features = importance_df.sort_values(by='importance', ascending=False).head(15)

    fig3 = plt.figure(figsize=(8, 4))
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title("Top 15 Features Influencing Issue Type")
    st.pyplot(fig3)

# Confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

if st.sidebar.checkbox("Show Confusion Matrices"):
    st.subheader("Confusion Matrix: Issue Type")
    plot_conf_matrix(y_test_issue, clf_issue.predict(X_test), "Issue Type")

    st.subheader("Confusion Matrix: Urgency Level")
    plot_conf_matrix(y_test_urgency, clf_urgency.predict(X_test), "Urgency Level")

# Classification reports
if st.sidebar.checkbox("Show Classification Reports"):
    st.subheader("Classification Report: Issue Type")
    report_issue = classification_report(y_test_issue, clf_issue.predict(X_test), output_dict=True)
    st.dataframe(pd.DataFrame(report_issue).transpose())

    st.subheader("Classification Report: Urgency Level")
    report_urgency = classification_report(y_test_urgency, clf_urgency.predict(X_test), output_dict=True)
    st.dataframe(pd.DataFrame(report_urgency).transpose())
