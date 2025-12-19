"""
LSTM Neural Network Modeling for Crude Oil Price Prediction

This script implements LSTM models with grid search hyperparameter tuning
for predicting crude oil prices. Uses the same data preparation as 
Time Series Modeling.ipynb.

Features:
- Grid search over multiple hyperparameters
- Top 10 models selected for ensemble prediction
- Results exported to CSV for further analysis
"""

# ============================================================================
# 1. Library Imports
# ============================================================================

import numpy as np
import pandas as pd
import warnings
import time
import os
from itertools import product

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("LSTM NEURAL NETWORK MODELING FOR CRUDE OIL PRICE PREDICTION")
print("=" * 70)

# ============================================================================
# 2. Data Preparation (Same as Time Series Modeling.ipynb)
# ============================================================================

print("\n[1/6] Loading and preparing data...")

# Load cleaned and shortened oil prediction data
df = pd.read_csv('../data/shortened_oil_data.csv')
print(f"Data shape: {df.shape}")

# Define target variable
TARGET = 'Crude_Oil'

# Key indicators based on domain knowledge (same as Time Series Modeling.ipynb)
KEY_INDICATORS = [
    # PADD 3 (Gulf Coast) - major refining hub
    'PADD3_Ref_NetIn_Crude',
    'PADD3_RefBl_NetProd_FinGas',
    'PADD3_Percent_Utilization_Refy_Operable_CapacityPct',
    'PADD3_Stocks_Ex_SPR_Crude',
    
    # Residual Fuel Oil
    'US_RefBl_NetProd_Residual',
    'US_Stocks_Residual',
    
    # US Crude Oil Stocks in Transit from Alaska
    'US_Crude_Stocks_Transit_from_AK',
    
    # Core market indicators
    'Brent_Oil',
    'DXY',
    'Gold',
    'Natural_Gas',
    'SP500',
    
    # Technical indicators
    'RSI_14',
    'MACD_Hist',
    'BB_Position',
    'SMA_20',
    'Realized_Vol_20d',
    
    # Lagged features
    'Oil_Lag1',
    'Oil_Lag2',
    'Oil_Lag5',
    'Ret_1d',
    'Ret_5d',
    
    # Supply/Demand fundamentals
    'US_Stocks_Crude',
    'US_FieldProd_Crude',
    'US_Imp_Crude',
    'US_Exp_Crude',
]

# Filter to only columns that exist in the dataframe
feature_cols = [col for col in KEY_INDICATORS if col in df.columns]
missing_cols = [col for col in KEY_INDICATORS if col not in df.columns]

print(f"Target: {TARGET}")
print(f"Selected key features: {len(feature_cols)}")
if missing_cols:
    print(f"Warning - Missing columns: {missing_cols}")

# Prepare data
X = df[feature_cols].values.astype(np.float64)
y = df[TARGET].values.astype(np.float64)

# Clean data - Replace NaN with column mean and Inf with large finite values
X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
y = np.nan_to_num(y, nan=np.nanmean(y), posinf=1e10, neginf=-1e10)

# Fill remaining NaN with column means
col_means = np.nanmean(X, axis=0)
col_means = np.nan_to_num(col_means, nan=0.0)
for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    X[mask, i] = col_means[i]

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Replace any remaining NaN/Inf after scaling
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================================
# 3. Sequence Creation for LSTM
# ============================================================================

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# 4. Define LSTM Model Builder
# ============================================================================

def build_lstm_model(seq_length, n_features, lstm_units, n_layers, dropout_rate, learning_rate):
    """Build and compile an LSTM model with specified hyperparameters."""
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(seq_length, n_features)))
    
    # LSTM layers
    for i in range(n_layers):
        return_sequences = (i < n_layers - 1)  # Only last LSTM layer returns single output
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ============================================================================
# 5. Grid Search Hyperparameter Tuning
# ============================================================================

print("\n[2/6] Setting up grid search hyperparameter space...")

# Define hyperparameter grid
param_grid = {
    'seq_length': [10, 15, 20],           # Sequence length
    'lstm_units': [32, 64, 128],          # LSTM units per layer
    'n_layers': [1, 2],                   # Number of LSTM layers
    'dropout_rate': [0.1, 0.2, 0.3],      # Dropout rate
    'learning_rate': [0.001, 0.0005],     # Learning rate
    'batch_size': [16, 32],               # Batch size
}

# Calculate total combinations
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)
print(f"Total hyperparameter combinations: {total_combinations}")

# Generate all combinations
param_combinations = list(product(
    param_grid['seq_length'],
    param_grid['lstm_units'],
    param_grid['n_layers'],
    param_grid['dropout_rate'],
    param_grid['learning_rate'],
    param_grid['batch_size'],
))

print(f"\nHyperparameter ranges:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# ============================================================================
# 6. Run Grid Search
# ============================================================================

print("\n[3/6] Running grid search (this may take a while)...")
start_time = time.time()

grid_results = []
n_features = X_scaled.shape[1]

# Training settings
EPOCHS = 100
PATIENCE = 10  # Early stopping patience

for idx, (seq_length, lstm_units, n_layers, dropout_rate, learning_rate, batch_size) in enumerate(param_combinations):
    
    # Progress update
    if (idx + 1) % 10 == 0 or idx == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {idx + 1}/{total_combinations} ({100*(idx+1)/total_combinations:.1f}%) - Elapsed: {elapsed:.1f}s")
    
    try:
        # Create sequences with current seq_length
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
        
        # Time-series split
        n_samples = len(X_seq)
        train_end = int(n_samples * TRAIN_RATIO)
        val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))
        
        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        
        # Build model
        model = build_lstm_model(seq_length, n_features, lstm_units, n_layers, dropout_rate, learning_rate)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        result = {
            'seq_length': seq_length,
            'lstm_units': lstm_units,
            'n_layers': n_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'epochs_trained': len(history.history['loss']),
        }
        grid_results.append(result)
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"  Error with params {idx + 1}: {e}")
        continue

elapsed_total = time.time() - start_time
print(f"\nGrid search completed in {elapsed_total:.1f} seconds")
print(f"Successfully evaluated {len(grid_results)} configurations")

# ============================================================================
# 7. Select Top 10 Models
# ============================================================================

print("\n[4/6] Selecting top 10 models by validation loss...")

# Convert to DataFrame and sort by validation loss
results_df = pd.DataFrame(grid_results)
results_df = results_df.sort_values('val_loss').reset_index(drop=True)

# Display top 10 models
print("\nTop 10 LSTM configurations:")
print(results_df.head(10).to_string())

# Get top 10 configurations
top_10_configs = results_df.head(10).to_dict('records')

# ============================================================================
# 8. Train Top 10 Models and Generate Predictions
# ============================================================================

print("\n[5/6] Training top 10 models and generating predictions...")

# Store predictions from each model
all_predictions = {}
model_metrics = []

# Use the sequence length from the best model for consistent test set
best_seq_length = top_10_configs[0]['seq_length']
X_seq_final, y_seq_final = create_sequences(X_scaled, y_scaled, best_seq_length)

n_samples = len(X_seq_final)
train_end = int(n_samples * TRAIN_RATIO)
val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train_final = X_seq_final[:train_end]
y_train_final = y_seq_final[:train_end]
X_val_final = X_seq_final[train_end:val_end]
y_val_final = y_seq_final[train_end:val_end]
X_test_final = X_seq_final[val_end:]
y_test_final = y_seq_final[val_end:]

print(f"Final test set size: {len(X_test_final)} samples")

for i, config in enumerate(top_10_configs):
    print(f"\n  Training Model {i+1}/10: units={config['lstm_units']}, layers={config['n_layers']}, "
          f"dropout={config['dropout_rate']}, lr={config['learning_rate']}, batch={config['batch_size']}")
    
    # Recreate sequences with this config's seq_length
    seq_length = config['seq_length']
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    n_samples = len(X_seq)
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    # Build and train model
    model = build_lstm_model(
        seq_length=seq_length,
        n_features=n_features,
        lstm_units=config['lstm_units'],
        n_layers=config['n_layers'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=config['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Generate predictions on test set
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform predictions to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Store predictions
    model_name = f"LSTM_Model_{i+1}"
    all_predictions[model_name] = y_pred
    
    # Calculate metrics for this model
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    model_metrics.append({
        'model_name': model_name,
        'seq_length': seq_length,
        'lstm_units': config['lstm_units'],
        'n_layers': config['n_layers'],
        'dropout_rate': config['dropout_rate'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
    })
    
    print(f"    Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Clear memory
    del model
    tf.keras.backend.clear_session()

# ============================================================================
# 9. Create Ensemble Prediction
# ============================================================================

print("\n[6/6] Creating ensemble prediction and exporting results...")

# Calculate ensemble prediction (average of top 10 models)
# Need to handle different sequence lengths - align predictions
ensemble_predictions = []

# For ensemble, use the predictions from each model
# Find the minimum test set size across all models
min_test_size = min(len(pred) for pred in all_predictions.values())

# Truncate all predictions to minimum size for ensemble
predictions_aligned = {}
for model_name, pred in all_predictions.items():
    predictions_aligned[model_name] = pred[-min_test_size:]

# Calculate ensemble (simple average)
ensemble_pred = np.mean(list(predictions_aligned.values()), axis=0)

# Get corresponding actual values (use the last model's y_actual aligned)
y_actual_aligned = y_actual[-min_test_size:]

# ============================================================================
# 10. Export Results to CSV
# ============================================================================

# Create results directory if it doesn't exist
results_dir = '../results'
os.makedirs(results_dir, exist_ok=True)

# Export 1: Grid search results
grid_search_path = os.path.join(results_dir, 'lstm_grid_search_results.csv')
results_df.to_csv(grid_search_path, index=False)
print(f"\nGrid search results saved to: {grid_search_path}")

# Export 2: Top 10 model metrics
metrics_df = pd.DataFrame(model_metrics)
metrics_path = os.path.join(results_dir, 'lstm_top10_model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Top 10 model metrics saved to: {metrics_path}")

# Export 3: Predictions from all top 10 models + ensemble
predictions_df = pd.DataFrame(predictions_aligned)
predictions_df['Ensemble_Prediction'] = ensemble_pred
predictions_df['Actual'] = y_actual_aligned
predictions_df.insert(0, 'Index', range(len(predictions_df)))

predictions_path = os.path.join(results_dir, 'lstm_predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

# Export 4: Summary statistics
summary_data = {
    'Metric': ['Total Models Evaluated', 'Top 10 Models Used', 'Test Set Size', 
               'Best Model RMSE', 'Best Model R²', 'Ensemble RMSE', 'Ensemble R²'],
    'Value': [
        len(grid_results),
        10,
        len(y_actual_aligned),
        metrics_df['test_rmse'].min(),
        metrics_df['test_r2'].max(),
        np.sqrt(mean_squared_error(y_actual_aligned, ensemble_pred)),
        r2_score(y_actual_aligned, ensemble_pred)
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(results_dir, 'lstm_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# ============================================================================
# 11. Final Summary
# ============================================================================

print("\n" + "=" * 70)
print("LSTM MODELING COMPLETE")
print("=" * 70)

print("\nBest Model Configuration:")
best_config = top_10_configs[0]
for key, value in best_config.items():
    print(f"  {key}: {value}")

print("\nTop 10 Model Performance Summary:")
print(metrics_df[['model_name', 'test_rmse', 'test_r2']].to_string(index=False))

ensemble_rmse = np.sqrt(mean_squared_error(y_actual_aligned, ensemble_pred))
ensemble_r2 = r2_score(y_actual_aligned, ensemble_pred)
print("\nEnsemble Prediction Performance:")
print(f"  RMSE: {ensemble_rmse:.4f}")
print(f"  R²: {ensemble_r2:.4f}")

print("\nOutput Files:")
print(f"  1. {grid_search_path}")
print(f"  2. {metrics_path}")
print(f"  3. {predictions_path}")
print(f"  4. {summary_path}")

print("\nAll results exported successfully!")

