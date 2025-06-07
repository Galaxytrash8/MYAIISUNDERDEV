import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load saved model
model = TFBertForSequenceClassification.from_pretrained('model/math_solver')
tokenizer = BertTokenizer.from_pretrained('model/math_solver')

# Test function
def solve_math_problem(problem_text):
    inputs = tokenizer(problem_text, 
                      return_tensors="tf",
                      padding=True, 
                      truncation=True)
    prediction = model(inputs).logits.numpy()[0][0]
    return round(prediction, 2)

# Test cases
test_problems = [
    "x + 3 = 7",
    "2 * x = 18",
    "x / 3 = 4"
]

for problem in test_problems:
    solution = solve_math_problem(problem)
    print(f"Problem: {problem} → Solution: {solution}")