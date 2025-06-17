import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_data(file_path='data/problems.json'):
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        # Validate data structure
        required_keys = ['problem', 'solution']
        for i, problem in enumerate(problems):
            if not all(key in problem for key in required_keys):
                raise ValueError(f"Missing required keys in problem {i}: {required_keys}")
        
        print(f"Loaded {len(problems)} problems from {file_path}")
        return problems
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please make sure the file exists.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data(problems, max_length=64):  # Reduced max_length for math problems
    """Prepare and tokenize the data"""
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Extract texts and labels
    texts = [p['problem'] for p in problems]
    solutions = [float(p['solution']) for p in problems]
    
    print("Tokenizing problems...")
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    
    labels = tf.convert_to_tensor(solutions, dtype=tf.float32)
    
    print(f"Dataset info:")
    print(f"  - Number of problems: {len(texts)}")
    print(f"  - Input shape: {inputs['input_ids'].shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Solution range: {min(solutions):.1f} to {max(solutions):.1f}")
    
    return inputs, labels, tokenizer

def create_splits(inputs, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):  # Adjusted splits
    """Create train/validation/test splits"""
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=42, shuffle=True
    )
    
    # Second split: val vs test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
    )
    
    # Create data splits
    def split_inputs(inputs_dict, indices):
        return {k: tf.gather(v, indices) for k, v in inputs_dict.items()}
    
    train_inputs = split_inputs(inputs, train_idx)
    val_inputs = split_inputs(inputs, val_idx)
    test_inputs = split_inputs(inputs, test_idx)
    
    train_labels = tf.gather(labels, train_idx)
    val_labels = tf.gather(labels, val_idx)
    test_labels = tf.gather(labels, test_idx)
    
    print(f"Data splits: Train={len(train_labels)}, Val={len(val_labels)}, Test={len(test_labels)}")
    
    return (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels)

def create_model():
    """Create and setup the model"""
    print("Loading BERT model...")
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=1
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=3e-5,  # Slightly higher learning rate
        weight_decay=0.01,
        epsilon=1e-8
    )
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    return model, optimizer, loss_fn

def train_model(model, optimizer, loss_fn, train_data, val_data, epochs=10, patience=3):  # Reduced epochs and patience
    """Train the model with early stopping"""
    train_inputs, train_labels = train_data
    val_inputs, val_labels = val_data
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training step
        with tf.GradientTape() as tape:
            outputs = model(train_inputs, training=True)
            train_loss = loss_fn(train_labels, tf.squeeze(outputs.logits))
        
        # Apply gradients
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Validation step
        val_outputs = model(val_inputs, training=False)
        val_loss = loss_fn(val_labels, tf.squeeze(val_outputs.logits))
        
        # Track losses
        train_losses.append(float(train_loss.numpy()))
        val_losses.append(float(val_loss.numpy()))
        
        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"Train Loss: {train_loss.numpy():.4f} | "
              f"Val Loss: {val_loss.numpy():.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                best_model_path = os.path.join('model', 'math_solver_best')
                os.makedirs(best_model_path, exist_ok=True)
                model.save_pretrained(best_model_path)
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            except Exception as e:
                print(f"  → Warning: Could not save best model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    print("-" * 60)
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses

# [Rest of the functions remain unchanged...]

def main():
    """Main execution function"""
    print("="*60)
    print("MATH EQUATION SOLVER TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs('model', exist_ok=True)
    
    # Load data from JSON file
    problems = load_data('data/problems.json')
    if problems is None:
        print("Failed to load data. Please check your data/problems.json file.")
        return
    
    # Prepare data
    inputs, labels, tokenizer = prepare_data(problems)
    
    # Create splits
    train_data, val_data, test_data = create_splits(inputs, labels)
    
    # Create model
    model, optimizer, loss_fn = create_model()
    
    # Train model
    train_losses, val_losses = train_model(
        model, optimizer, loss_fn, train_data, val_data,
        epochs=10, patience=3  # Updated training parameters
    )
    
    # [Rest of the main function remains unchanged...]

if __name__ == "__main__":
    main()
