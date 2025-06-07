import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# 1. Préparation des données
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

problems = [
    ("x + 3 = 7", 4),
    ("2 * x = 10", 5),
    ("x^2 = 16", 4),
    ("x / 2 = 3", 6),
    ("3x - 5 = 10", 5)
]

inputs = tokenizer([p[0] for p in problems], padding=True, truncation=True, return_tensors="tf")
labels = tf.convert_to_tensor([p[1] for p in problems], dtype=tf.float32)

# 2. Modèle
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=1
)

# 3. Entraînement
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
loss_fn = tf.keras.losses.MeanSquaredError()

for epoch in range(10):  # 10 epochs
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = loss_fn(labels, tf.squeeze(outputs.logits))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# 4. Sauvegarde
model.save_pretrained('model/saved_model')
tokenizer.save_pretrained('model/saved_model')

# 5. Test
test_problems = [
    "x + 7 = 10",  # Doit prédire ~3
    "4 * x = 20"   # Doit prédire ~5
]

test_inputs = tokenizer(test_problems, padding=True, truncation=True, return_tensors="tf")
predictions = model(test_inputs).logits

for problem, pred in zip(test_problems, predictions.numpy()):
    print(f"Problème: {problem}, Prédiction: {pred[0]:.1f}")