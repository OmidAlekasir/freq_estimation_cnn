import numpy as np
from scipy.io import savemat, loadmat
from utils import data_to_tensors, create_model
from sklearn.model_selection import train_test_split

### @@@ DATA PREPARATION @@@ ###
### Import data ###
path = '../../MSc/matlab_codes/patients_dataset'
data = loadmat(path)

### convert data to tensors for CNN model ###
X, Y = data_to_tensors(data)
X = np.array(X).astype(np.float32)
Y = np.array(Y)

savemat('radar_data_cnn.mat', {'radar': X, 'freq': Y})

### Normalize ###
bpm_max = 180
X /= np.max(np.abs(X)) # normalize the data for CNN
Y /= bpm_max # normalize the targets for CNN (max bpm: 180)

### Train-Test split ###
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

### @@@ Create and compile the model @@@ ###
input_shape = (X.shape[1], X.shape[2])
model = create_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')

### @@@ FIT THE MODEL @@@ ###
history = model.fit(X_train, Y_train, epochs=35, batch_size=64, validation_split=0.1)

### @@@ TEST THE MODEL @@@ ###
prd = model.predict(X)
prd_test = model.predict(X_test)

E1 = Y - prd.T
E2 = Y_test - prd_test.T
print('Train MAE:', bpm_max * np.mean(np.abs(E1)))
print('Test MAE:', bpm_max * np.mean(np.abs(E2)))

### Save data for plotting ###
np.savez('plot_data.npz',
         loss=history.history['loss'],
         val_loss=history.history['val_loss'],
         Y=Y, prd=prd, Y_test=Y_test, prd_test=prd_test)

model.save('model.h5')