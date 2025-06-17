import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load saved model
model = TFBertForSequenceClassification.from_pretrained('model/math_solver_best')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Test function
def solve_math_problem(problem_text):
    inputs = tokenizer(problem_text,
                       return_tensors="tf",
                       padding=True,
                       truncation=True)
    prediction = model(inputs).logits.numpy()[0][0]
    return round(prediction, 2)

# Load test data from JSON file
def load_test_data(file_path='data/problems.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # Use last 15% as test data (same as training split)
    test_size = int(len(problems) * 0.15)
    test_problems = problems[-test_size:]
    
    return test_problems

# Evaluate model on test data
def evaluate_model():
    print("Loading test data...")
    test_data = load_test_data()
    
    predictions = []
    true_values = []
    
    print(f"Evaluating {len(test_data)} test problems...\n")
    
    for i, item in enumerate(test_data):
        problem = item['problem']
        true_solution = float(item['solution'])
        predicted_solution = solve_math_problem(problem)
        
        predictions.append(predicted_solution)
        true_values.append(true_solution)
        
        # Show first 10 examples
        if i < 10:
            error = abs(predicted_solution - true_solution)
            print(f"Problem: {problem}")
            print(f"True: {true_solution}, Predicted: {predicted_solution}, Error: {error:.2f}")
            print("-" * 50)
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate accuracy within different thresholds
    errors = np.abs(np.array(true_values) - np.array(predictions))
    acc_01 = np.mean(errors <= 0.1) * 100
    acc_05 = np.mean(errors <= 0.5) * 100
    acc_10 = np.mean(errors <= 1.0) * 100
    
    print(f"\nEVALUATION RESULTS:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Accuracy within ±0.1: {acc_01:.1f}%")
    print(f"Accuracy within ±0.5: {acc_05:.1f}%")
    print(f"Accuracy within ±1.0: {acc_10:.1f}%")

# Test cases for manual testing
test_problems = [
    "x + 5 = 0",
    "x + 3 = 9",
    "2x = 16",
    "x - 4 = 6",
    "x/3 = 4",
    "3x + 5 = 14",
    "x + 7 = 15",
    "5x = 25",
    "x - 8 = -2",
    "x/2 = 8"
]

print("MANUAL TEST CASES:")
print("=" * 40)
for problem in test_problems:
    solution = solve_math_problem(problem)
    print(f"Problem: {problem} → Solution: {solution}")

print("\n" + "=" * 40)
print("AUTOMATED EVALUATION ON TEST DATA:")
print("=" * 40)

# Run evaluation on test dataset
evaluate_model()
