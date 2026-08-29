from pathlib import Path
import os
import tensorflow as tf
from data_loader import SunRGBDGenerator
from model_builder import build_efficient_unet

# --- CONFIGURATION ---
BATCH_SIZE = 8   #Can be raised or lowered depending ongpu memory, for my 3080ti 8 works well
EPOCHS = 30
LR = 1e-4        # Learning Rate

# Paths relativos a la raiz del proyecto (EDITAR si tu dataset esta en otro lado)
# Antes: C:\Users\victo\OneDrive\Documents\TT\SUNRBG_IMAGES\...
ROOT = Path(__file__).resolve().parents[2]
TRAIN_RGB   = str(ROOT / "data" / "SUNRGBD" / "Train" / "rgb")
TRAIN_INPUT = str(ROOT / "data" / "SUNRGBD" / "Train" / "depth_input")
TRAIN_GT    = str(ROOT / "data" / "SUNRGBD" / "Train" / "depth_gt")
TRAIN_MASK  = str(ROOT / "data" / "SUNRGBD" / "Train" / "mask")

VAL_RGB     = str(ROOT / "data" / "SUNRGBD" / "Validation" / "rgb")
VAL_INPUT   = str(ROOT / "data" / "SUNRGBD" / "Validation" / "depth_input")
VAL_GT      = str(ROOT / "data" / "SUNRGBD" / "Validation" / "depth_gt")
VAL_MASK    = str(ROOT / "data" / "SUNRGBD" / "Validation" / "mask")

# Preparing Data Generators
print("Creating Data Generators...")
train_gen = SunRGBDGenerator(TRAIN_RGB, TRAIN_INPUT, TRAIN_MASK, TRAIN_GT, batch_size=BATCH_SIZE)
val_gen = SunRGBDGenerator(VAL_RGB, VAL_INPUT, VAL_MASK, VAL_GT, batch_size=BATCH_SIZE)

# Build model with 5 channels
#This is because we are using RGB + Depth Input (1 channel) + Mask (1 channel) = 5 channels, EfficientNet expects 3 channels normally
print("Building EfficientNet U-Net...")
model = build_efficient_unet(input_shape=(480, 640, 5))

#Build 

# Compile
# Optimizer: Adam is standard.
# Loss: Mean Squared Error (MSE) is good for regression.
# Metric: Root Mean Squared Error (RMSE) is the scientific standard for depth.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='mean_squared_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Callbacks for saving the best model and early stopping
checkpoint_path = "best_depth_model.keras"

callbacks = [
    # Save the model only when validation loss improves
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
    
    # Stop if it stops learning after 5 epochs
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    
    # Reduce learning rate if it gets stuck
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# Training
print("Starting Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Training finished. Model saved as:", checkpoint_path)

