import numpy as np
from tensorflow.keras import layers, models

def data_to_tensors(data):
    ### @@@ gather data @@@ ###
    # reference data (time & freq)
    time_freq = data['time_freq'].squeeze() # time of reference freq.
    freq = data['freq'].squeeze() # reference freq

    # radar data
    time_radar = data['time_radar'].squeeze() # time of radar data
    I = data['I'].squeeze() # time of radar data
    Q = data['Q'].squeeze() # time of reference freq.

    # # reference signal
    # time_ref = data['time_ref'].squeeze() # reference freq
    # ref = data['ref'].squeeze() # reference freq

    X = None
    number_of_patients = len(I)

    for i in range(number_of_patients):
        
        ### @@@ single patient data @@@ ###
        tr = time_radar[i].squeeze()
        tf = time_freq[i].squeeze()
        I0 = I[i].squeeze()
        Q0 = Q[i].squeeze()
        freq0 = freq[i].squeeze()

        x, y = __signal_segmentation(tr, tf, I0, Q0, freq0)
        print(f'Patient number {i} loaded.')

        if X is None:
            X = x
            Y = y
            continue

        X = np.vstack((X, x))
        Y = np.hstack((Y, y))

    return X, Y

def __signal_segmentation(time_data, time_freq, I, Q, freq):

    ### @@@ Preparation & Segmentation @@@ ###
    # Compute data freq.
    dt = time_data[1] - time_data[0]
    f = int(1/60/dt)

    stride = 2 * f # ~ 2 seconds
    len_segment = 30 * f # ~ 30 seconds
    N = len(time_data)

    X = []
    Y = []

    for i in range(N // stride - 20):
        bias = i * stride

        # define input
        time_segment = time_data[bias : bias + len_segment]

        segment1 = I[bias : bias + len_segment] # I
        segment2 = Q[bias : bias + len_segment] # Q
        segment = np.array([segment1, segment2]).T

        # define target
        idx0 = np.where(time_freq >= time_segment[0])[0][0] # first sample index of target freq.
        idx1 = np.where(time_freq >= time_segment[-1])[0][0] # last sample index of target freq.
        target = np.mean(freq[idx0:idx1]) # mean target freq.

        # Gather train/test data
        X.append(list(segment))
        Y.append(target)

    return X, Y

# Define the CNN model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
        layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
        layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Output layer: estimated frequency
    ])
    return model