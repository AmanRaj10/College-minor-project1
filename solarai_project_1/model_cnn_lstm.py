import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def build_cnn_lstm(input_shape, dropout_rate=0.15):
    """
    Enhanced CNN-BiLSTM-Attention model for solar power forecasting.
    
    Architecture:
    - Multi-scale CNN (kernel 3, 5, 7) for local pattern extraction
    - Bidirectional LSTM ×2 for temporal dependencies
    - Multi-head Self-Attention for global context
    - Residual connections to prevent gradient vanishing
    - Output head with progressive narrowing
    
    Target accuracy: 88–95% (daytime MAPE-based)
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')

    # ── Multi-scale CNN feature extraction ──────────────
    conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv_k3')(inputs)
    conv5 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', name='conv_k5')(inputs)
    conv7 = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', name='conv_k7')(inputs)
    conv1 = layers.Conv1D(32, kernel_size=1, padding='same', activation='relu', name='conv_k1')(inputs)  # pointwise

    merged = layers.Concatenate(name='cnn_merge')([conv3, conv5, conv7, conv1])
    merged = layers.BatchNormalization(name='bn_cnn')(merged)
    merged = layers.Dropout(dropout_rate, name='drop_cnn')(merged)

    # ── Residual block ────────────────────────────────────
    res = layers.Conv1D(192, 1, padding='same', name='res_proj')(merged)  # project for residual
    merged = layers.MaxPooling1D(pool_size=2, name='maxpool')(merged)

    # ── Bidirectional LSTM stack ─────────────────────────
    lstm1 = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=dropout_rate/2, recurrent_dropout=0.0),
        name='bilstm1'
    )(merged)
    lstm1 = layers.Dropout(dropout_rate, name='drop_lstm1')(lstm1)

    lstm2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=dropout_rate/2, recurrent_dropout=0.0),
        name='bilstm2'
    )(lstm1)
    lstm2 = layers.Dropout(dropout_rate, name='drop_lstm2')(lstm2)

    lstm3 = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=dropout_rate/2),
        name='bilstm3'
    )(lstm2)

    # ── Multi-head Self-Attention ─────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, name='mhsa')(lstm3, lstm3)
    attn = layers.LayerNormalization(name='ln_attn')(attn + lstm3)

    # ── Temporal attention (squeeze-excite style) ─────────
    gate = layers.GlobalAveragePooling1D(name='gate_avg')(attn)
    gate = layers.Dense(32, activation='relu', name='gate_fc1')(gate)
    gate = layers.Dense(attn.shape[-1], activation='sigmoid', name='gate_fc2')(gate)
    gate = layers.Reshape((1, attn.shape[-1]), name='gate_reshape')(gate)
    attn_gated = layers.Multiply(name='gate_mul')([attn, gate])

    # ── Pooling ───────────────────────────────────────────
    avg_pool = layers.GlobalAveragePooling1D(name='global_avg')(attn_gated)
    max_pool = layers.GlobalMaxPooling1D(name='global_max')(attn_gated)
    pooled = layers.Concatenate(name='pool_merge')([avg_pool, max_pool])

    # ── Output head ──────────────────────────────────────
    x = layers.Dense(256, activation='relu', name='fc1')(pooled)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Dropout(dropout_rate, name='drop_fc1')(x)

    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate / 2, name='drop_fc2')(x)

    x = layers.Dense(64, activation='relu', name='fc3')(x)
    x = layers.Dropout(dropout_rate / 4, name='drop_fc3')(x)

    # Final output — clamp with relu for non-negative solar power
    output = layers.Dense(1, activation='relu', name='output')(x)

    model = Model(inputs, output, name='CNN_BiLSTM_Attention_v2')
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=5e-4,
            weight_decay=1e-5
        ),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


def build_callbacks(model_path='saved_model/cnn_lstm_model.keras'):
    """Callbacks for training with aggressive LR scheduling."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            monitor='val_loss',
            verbose=1,
            cooldown=3
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv', append=False),
        tf.keras.callbacks.TerminateOnNaN()
    ]