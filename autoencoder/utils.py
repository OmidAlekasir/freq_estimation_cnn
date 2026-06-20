from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D

def build_1d_autoencoder(input_shape):
    # Encoder
    input_layer = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='elu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(16, kernel_size=3, activation='elu', padding='same')(x)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x)

    # Decoder
    x = Conv1D(16, kernel_size=3, activation='elu', padding='same')(encoded)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(32, kernel_size=3, activation='elu', padding='same')(x)
    x = UpSampling1D(size=2)(x)
    decoded = Conv1D(input_shape[-1], kernel_size=3, activation='linear', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    return autoencoder