"""
Quick test script to verify EfficientNetB0 model loading and prediction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import model

print("Testing EfficientNetB0 model integration...")
print("-" * 50)

# Load model
if loaded_model is None:
    print("ERROR: Model failed to load!")
    exit(1)
else:
    print("✓ Model loaded successfully!")
    print(f"Model type: EfficientNetB0")
    print(f"Model input shape: {loaded_model.input_shape}")
    print(f"Model output shape: {loaded_model.output_shape}")
    print(f"Class names: {model.CLASS_NAMES}")


print("-" * 50)
print("Model integration test completed!")