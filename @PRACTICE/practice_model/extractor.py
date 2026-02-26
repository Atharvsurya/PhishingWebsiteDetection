import pandas as pd
import os
import joblib

# 1. Load your frozen model
model = joblib.load('PRACTICE/practice_model/random_forest_v1.pkl')
log_file = 'prediction_history.csv'

def predict_and_recall(url):
    # --- PART A: RECALL (Check Memory) ---
    if os.path.exists(log_file):
        history = pd.read_csv(log_file)
        # Check if we've seen this URL before
        if url in history['url'].values:
            past_result = history[history['url'] == url]['label'].values[0]
            return f"RECALLED: We already know this site is {past_result}."

    # --- PART B: PREDICT (New Knowledge) ---
    # Convert URL to features (assuming 'len' and 'dots')
    features = pd.DataFrame([[len(url), url.count('-')]], columns=['len', 'dash'])
    prediction = model.predict(features)[0]
    result = "PHISHING" if prediction == 1 else "SAFE"

    # --- PART C: STORE (Save to Memory) ---
    new_entry = pd.DataFrame([[url, result]], columns=['url', 'label'])
    # Append to CSV without overwriting
    new_entry.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    return f"NEW PREDICTION: This site is {result}. Saved to memory."

# Test it
print(predict_and_recall("verify-login-bank.com"))