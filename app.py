from flask import Flask, render_template, request, jsonify
import pandas as pd
import spacy
from rapidfuzz import process, fuzz

# Initialize Flask app
app = Flask(__name__)

# Load the CSV data and SpaCy model
master_data = pd.read_csv(r"C:\Users\ASUS\Downloads\cleaned_master_data.csv")
nlp = spacy.load('en_core_web_sm')

# Check if the 'Cleaned Medicine Name' column is in the dataset
if 'Cleaned Medicine Name' not in master_data.columns:
    raise ValueError("The 'Cleaned Medicine Name' column is missing from the CSV file")

# Prepare list of medicine names in lowercase
medicine_names = master_data['Cleaned Medicine Name'].astype(str).str.lower().tolist()

def extract_medicine_with_fuzzy(text, threshold=80):
    text = text.lower()
    found_medicines = []

    # Tokenize text while keeping relevant medical terms only
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and len(token.text) > 6]  # Ignore short tokens

    # Match each token against the medicine names using fuzzy matching
    for token in tokens:
        result = process.extractOne(token, medicine_names, scorer=fuzz.ratio)
        if result:  # Ensure a match was found
            match, score, _ = result  # Extract the match and score
            if score >= threshold:  # Consider it a match if similarity is above the threshold
                found_medicines.append(match.capitalize())

    # Filter to include only medicines in the original list and remove duplicates
    valid_medicines = sorted(set(med for med in found_medicines if med.lower() in medicine_names))
    return {"medications": valid_medicines}

# Define route for the main page
@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML form for user input

# Define endpoint to process the input and extract medicines
@app.route('/extract_medicines', methods=['POST'])
def extract_medicines():
    if request.method == 'POST':
        patient_prompt = request.form['prompt']  # Get the input from the form
        extracted_meds = extract_medicine_with_fuzzy(patient_prompt)
        return jsonify(extracted_meds)  # Return the result as JSON

if __name__ == '__main__':
    app.run(debug=True)
