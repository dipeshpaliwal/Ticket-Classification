## 📂 Task 1 – Ticket Classification & Entity Extraction

### 🧠 Objective

Build a machine learning pipeline that:
- Classifies support tickets by **issue type** and **urgency level**
- Extracts structured entities like product names, complaint keywords, and dates from ticket text

---

### 📁 Dataset

File: `ai_dev_assignment_tickets_complex_1000.xls`  
Columns:
- `ticket_id`
- `ticket_text`
- `issue_type` *(label)*
- `urgency_level` *(label: Low, Medium, High)*
- `product` *(ground truth for entity extraction)*

---

### ⚙️ Features Implemented

- Data cleaning: lowercasing, removing special characters, stopword removal
- Feature Engineering: TF-IDF, ticket length, sentiment score
- Multi-class classification:
  - **Issue Type Classifier**
  - **Urgency Level Classifier**
- Rule-based Named Entity Extraction:
  - Product names (regex or keyword match)
  - Complaint keywords
  - Dates
- **Streamlit App** for interactive ticket predictions
- Visualizations:
  - Ticket class distributions
  - Confusion matrices
- Batch processing support

---

### ▶️ Run the App

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run Streamlit app
streamlit run ticket_classifier_app.py
