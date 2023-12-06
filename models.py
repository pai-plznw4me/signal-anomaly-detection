from keras.models import Model
from keras.layers import Input, LSTM, Dense, BatchNormalization, LayerNormalization
import pandas as pd
import numpy as np
from copy import copy
from sklearn import preprocessing


def create_lstm_autoencoder(input_steps, input_n_features, hidden_units):
    """
    LSTM 오토인코더 모델을 생성하는 함수.

    Parameters:
    - input_steps (int): 입력 시퀀스의 타임 스텝 수
    - input_n_features (int): 입력 시퀀스의 각 타임 스텝에서의 피처 수
    - hidden_units (int): LSTM 레이어의 은닉 유닛 수

    Returns:
    - model (tf.keras.Model): 생성된 LSTM 오토인코더 모델

    Example:
    >>> input_steps = 10
    >>> input_n_features = 32
    >>> hidden_units = 64
    >>> autoencoder_model = create_lstm_autoencoder(input_steps, input_n_features, hidden_units)
    """
    # 인코더 정의
    encoder_inputs = Input(shape=(input_steps, input_n_features))
    encoder_lstm, state_h, state_c = LSTM(hidden_units, return_state=True, return_sequences=True)(encoder_inputs)
    encoder_states = [state_h, state_c]
    code = encoder_lstm

    # 디코더 정의
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(encoder_lstm, initial_state=encoder_states)
    decoder_dense = Dense(input_n_features, activation=None)
    decoder_outputs = decoder_dense(decoder_outputs)

    # 모델 정의
    model = Model(encoder_inputs, decoder_outputs)
    return model


def anomaly_classification(input_steps, input_n_features, hidden_units):
    input_ = Input(shape=(input_steps, input_n_features))
    x = input_
    x = LSTM(hidden_units)(x)
    x = Dense(units=hidden_units, activation='tanh')(x)
    x = Dense(units=hidden_units, activation='tanh')(x)
    output = Dense(units=2, activation='softmax')(x)
    model = Model(input_, output)
    return model



