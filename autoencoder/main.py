import numpy as np
from scipy.io import loadmat, savemat
from utils import build_1d_autoencoder

from sklearn.model_selection import train_test_split

# Load data #
path = '../STFT_NMF/radar_data_autoencoder'
data_raw = loadmat(path)

data = data_raw['data']

X = data[:, :, :2] # noisy data
Y = data[:, :, 2:] # clean data

X = X / np.max(np.abs(X))
Y = Y / np.max(np.abs(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate and compile the model
input_shape = (X.shape[1], X.shape[2])
model = build_1d_autoencoder(input_shape)
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, Y_train,
                epochs=8,
                batch_size=64,
                validation_split=0.1)
model.save('autoencoder.h5')

# Ensure the loss history keys are correctly saved
loss_history = {
    'train_loss': model.history.history['loss'],
    'val_loss': model.history.history['val_loss']
}
savemat('loss_history.mat', loss_history)

### @@@ TEST THE MODEL @@@ ###
prd = model.predict(X)
prd_test = model.predict(X_test)

# savemat('prd_all.mat', {'actual': X, 'denoised': Y, 'prediction': prd})
savemat('prd_test.mat', {'actual': X_test, 'denoised': Y_test, 'prediction': prd_test})