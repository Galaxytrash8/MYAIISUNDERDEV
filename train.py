import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# 1. Load data
with open('data/problems.json') as f:
    problems = json.load(f)

# 2. Prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
inputs = tokenizer([p['problem'] for p in problems], 
                  padding=True, 
                  truncation=True, 
                  return_tensors="tf")
labels = tf.convert_to_tensor([p['solution'] for p in problems], 
                             dtype=tf.float32)

# 3. Initialize model
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=1
)

# 4. Training setup
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
loss_fn = tf.keras.losses.MeanSquaredError()

# 5. Training loop
for epoch in range(10):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = loss_fn(labels, tf.squeeze(outputs.logits))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# 6. Save model
model.save_pretrained('model/math_solver')
tokenizer.save_pretrained('model/math_solver')