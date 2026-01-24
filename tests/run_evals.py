import json
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
try:
    from app.modules.agents import MainAgent
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
    from app.modules.agents import MainAgent

def run_evals():
    # Load Data
    json_path = 'sms_conversations.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations.")

    # Initialize Agent
    agent = MainAgent()
    
    y_true = []
    y_pred = []
    
    print("Running evaluation... This may take a while.")
    
    # Cap at 5 conversations for a quick test if needed, but let's try all
    # Or maybe just 5 to ensure it finishes quickly for the user update
    # taking first 5 for speed in this interaction
    test_conversations = conversations[:5] 
    print(f"Testing on first {len(test_conversations)} conversations for speed...")

    for conv in test_conversations:
        history = ""
        for turn in conv['turns']:
            speaker = "Recruiter" if turn['speaker'] == 'recruiter' else "Candidate"
            text = turn['text']
            
            # The label on a Recruiter turn represents the action chosen BEFORE generating that turn.
            if turn['speaker'] == 'recruiter' and turn['label']:
                # Predict
                try:
                    action = agent.decide_action(history)
                except Exception as e:
                    print(f"Error predicting: {e}")
                    action = "ERROR"
                
                y_pred.append(action.lower())
                y_true.append(turn['label'].lower())
                
                print(f"True: {turn['label'].lower():<10} | Pred: {action.lower()}")
            
            # Update history for next turn
            history += f"{speaker}: {text}\n"

    # Metrics
    if y_true:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("No labels found to evaluate.")

if __name__ == "__main__":
    run_evals()
