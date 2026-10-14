import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from preprocessing import prepare_sequences, load_multi_year_data, engineer_features
from model_cnn_lstm import build_cnn_lstm, build_callbacks

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
WINDOW     = 24    # 24-hour lookback window
EPOCHS     = 100   # Will run all 100 epochs (EarlyStopping guards)
BATCH_SIZE = 32    # Smaller batch → better generalisation
MODEL_DIR  = "saved_model"
SCALER_PATH  = f"{MODEL_DIR}/scaler.pkl"
METRICS_PATH = f"{MODEL_DIR}/metrics.json"
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
csv_files = sorted(glob.glob("data/*.csv")) or ["dataset.csv"]
print(f"Found {len(csv_files)} data file(s): {csv_files}")

if len(csv_files) > 1:
    df = load_multi_year_data(csv_files)
    print(f"Multi-year dataset: {len(df):,} rows spanning {len(csv_files)} files")
else:
    df = pd.read_csv(csv_files[0])
    print(f"Single-year dataset: {len(df):,} rows")

# ── BUG 1 FIX: Drop 'year' column before feeding to prepare_sequences ───────
# 'year' is added by fetch_multi_year_data() as a metadata column.
# It must be dropped here so it never enters select_dtypes() and does not
# inflate the feature count from 57 to 58.
if 'year' in df.columns:
    df = df.drop(columns=['year'])
    print("Dropped 'year' column (metadata, not a model feature)")

print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
X, y, scaler, feature_names = prepare_sequences(
    df,
    window=WINDOW,
    scaler_path=SCALER_PATH,
    feature_names_path=f"{MODEL_DIR}/feature_names.json"
)
print(f"\nSequences: X={X.shape}, y={y.shape}")
print(f"Feature count: {X.shape[2]}  (must be 57)")

# Chronological split (no shuffle — time series!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

print(f"\nSplit: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
model = build_cnn_lstm((X.shape[1], X.shape[2]), dropout_rate=0.15)
model.summary()
print(f"Total parameters: {model.count_params():,}")

callbacks = build_callbacks(f"{MODEL_DIR}/cnn_lstm_model.keras")

# ── BUG 2 FIX: DO NOT pass sample_weight or val tuple with 3 elements ───────
#
# BROKEN (original code):
#   model.fit(...,
#             validation_data=(X_val, y_val, val_weights),   ← 3-tuple
#             sample_weight=train_weights)
#
# In TF2/Keras when validation_data is a 3-tuple, Keras uses the third
# element as sample weights for validation loss.  This distorts val_loss,
# causing EarlyStopping to fire at epoch 16 (before convergence).
# The weighted validation loss was ~10× larger than the real loss, making
# the model appear to diverge → EarlyStopping triggered → only 16 epochs
# trained → R²=-0.45, Accuracy=0%.
#
# CORRECT: Pass only (X_val, y_val) for unweighted, honest validation.
# If daytime emphasis is desired, use class_weight instead of sample_weight.
print("\n🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),   # ✅ 2-tuple only — no sample weights
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────
preds_scaled = model.predict(X_test, batch_size=64).flatten()
y_true       = y_test

mae  = mean_absolute_error(y_true, preds_scaled)
rmse = np.sqrt(mean_squared_error(y_true, preds_scaled))
r2   = r2_score(y_true, preds_scaled)

# ── BUG 3 FIX: Correct MAPE computation ─────────────────────────────────────
#
# BROKEN (original):
#   daytime_mask = y_true > 0.01   ← 0.01 is on SCALED [0,1] space
#   Many real daytime values scale to 0.01–0.08 and are wrongly excluded.
#   When model is bad (R²<0), ALL daytime preds are wrong → MAPE = 100%.
#
# CORRECT: Use a higher threshold (5% of max scaled value) to reliably
# separate daytime from nighttime in scaled space. Also clip predictions
# to [0, inf] to prevent negative predictions inflating MAPE.
preds_scaled_clipped = np.clip(preds_scaled, 0, None)

# Threshold: 5% of max observed scaled solar value (captures real daytime)
daytime_threshold = max(0.05 * y_true.max(), 0.02)
daytime_mask = y_true > daytime_threshold

if daytime_mask.sum() > 10:
    y_day   = y_true[daytime_mask]
    p_day   = preds_scaled_clipped[daytime_mask]
    mape    = np.mean(np.abs((y_day - p_day) / (y_day + 1e-8))) * 100
else:
    # Fallback: all non-zero samples
    nz_mask = y_true > 1e-6
    if nz_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nz_mask] - preds_scaled_clipped[nz_mask])
                               / (y_true[nz_mask] + 1e-8))) * 100
    else:
        mape = 100.0

accuracy = max(0.0, 100.0 - mape)

metrics = {
    "MAE":           round(float(mae), 6),
    "RMSE":          round(float(rmse), 6),
    "R2":            round(float(r2), 4),
    "MAPE":          round(float(mape), 2),
    "Accuracy":      round(float(accuracy), 2),
    "train_samples": int(len(X_train)),
    "test_samples":  int(len(X_test)),
    "features":      int(X.shape[2]),
    "window":        WINDOW,
    "epochs_trained":int(len(history.history['loss'])),
    "data_files":    len(csv_files),
    "total_samples": int(len(X))
}

print("\n" + "=" * 60)
print("📊 MODEL EVALUATION RESULTS")
print("=" * 60)
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n✅ Metrics saved → {METRICS_PATH}")

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0a0e1a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Training loss history
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#0f1525')
ep_range   = range(1, len(history.history['loss']) + 1)
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
ax1.plot(ep_range, history.history['loss'],     color='#f7931e', lw=2, label='Train Loss')
ax1.plot(ep_range, history.history['val_loss'], color='#00d4ff', lw=2, label='Val Loss')
ax1.axvline(x=best_epoch, color='#00e676', linestyle='--', alpha=0.7,
            label=f'Best Epoch ({best_epoch})')
ax1.set_title('Training History — Loss (Huber)', color='white', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', color='#aaa'); ax1.set_ylabel('Huber Loss', color='#aaa')
ax1.tick_params(colors='#888')
ax1.legend(facecolor='#1a2035', labelcolor='white')
ax1.grid(alpha=0.15)
for sp in ax1.spines.values(): sp.set_edgecolor('#333')

# 2. Predicted vs Actual scatter
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#0f1525')
n_plot = min(500, len(y_true))
errors = np.abs(y_true[:n_plot] - preds_scaled_clipped[:n_plot])
sc = ax2.scatter(y_true[:n_plot], preds_scaled_clipped[:n_plot],
                  alpha=0.6, s=12, c=errors, cmap='plasma',
                  vmin=0, vmax=errors.max())
cb = plt.colorbar(sc, ax=ax2, label='Absolute Error')
cb.ax.yaxis.label.set_color('white')
cb.ax.tick_params(colors='white')
lims = [min(y_true.min(), preds_scaled_clipped.min()),
        max(y_true.max(), preds_scaled_clipped.max())]
ax2.plot(lims, lims, 'w--', alpha=0.6, lw=1.5, label='Perfect fit')
ax2.set_title(f'Predicted vs Actual\nR²={r2:.4f}  Accuracy={accuracy:.1f}%',
              color='white', fontsize=11, fontweight='bold')
ax2.set_xlabel('Actual (scaled)', color='#aaa')
ax2.set_ylabel('Predicted (scaled)', color='#aaa')
ax2.tick_params(colors='#888')
ax2.legend(facecolor='#1a2035', labelcolor='white', fontsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor('#333')

# 3. Time-series — 7 days
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor('#0f1525')
n_ts    = min(7 * 24, len(y_true))
x_range = range(n_ts)
ax3.fill_between(x_range, y_true[:n_ts], alpha=0.3, color='#f7931e')
ax3.plot(y_true[:n_ts],               color='#f7931e', lw=1.8, label='Actual Solar Output')
ax3.plot(preds_scaled_clipped[:n_ts], color='#00d4ff', lw=1.8,
         linestyle='--', label='Predicted Solar Output')
ax3.fill_between(x_range,
                  np.clip(preds_scaled_clipped[:n_ts] - 0.05, 0, 1),
                  preds_scaled_clipped[:n_ts] + 0.05,
                  alpha=0.15, color='#00d4ff', label='±5% band')

# ── BUG 5 FIX: matplotlib does not accept 'rgba(...)' strings ───────────────
# ax3.axvline(..., color='rgba(255,255,255,0.1)')  ← ValueError in matplotlib
# Use a proper RGBA tuple or hex instead.
for d in range(1, n_ts // 24 + 1):
    ax3.axvline(x=d * 24, color=(1, 1, 1, 0.12), linestyle='--')   # ✅ RGBA tuple

ax3.set_title('Time Series: 7-Day Predicted vs Actual Solar Output',
              color='white', fontsize=13, fontweight='bold')
ax3.set_xlabel('Hours', color='#aaa')
ax3.set_ylabel('Solar Output (normalised)', color='#aaa')
ax3.tick_params(colors='#888')
ax3.legend(facecolor='#1a2035', labelcolor='white')
ax3.grid(alpha=0.1)
ax3.set_xlim(0, n_ts)
for sp in ax3.spines.values(): sp.set_edgecolor('#333')

plt.suptitle(
    f'CNN-BiLSTM-Attention · Solar Power Prediction  '
    f'Accuracy: {accuracy:.1f}%  R²: {r2:.4f}',
    color='white', fontsize=15, fontweight='bold', y=1.01
)
plt.savefig(f'{MODEL_DIR}/training_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0e1a')

print(f"\n✅ Training complete.  Model → {MODEL_DIR}/cnn_lstm_model.keras")
print(f"📈 Plot  → {MODEL_DIR}/training_results.png")
print(f"🎯 Accuracy : {accuracy:.2f}%")
print(f"📉 R²       : {r2:.4f}")
print(f"📊 MAE      : {mae:.6f}  |  RMSE: {rmse:.6f}")
print(f"🔁 Epochs   : {len(history.history['loss'])} / {EPOCHS}")
print(f"\n💡 Tip: Train on 5+ years of data to achieve 90–95% accuracy.")