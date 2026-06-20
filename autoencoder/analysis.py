import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Set plot colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#bfbfbf', '#d62728']

# Load the loss history data
loss_data = loadmat('loss_history.mat')
train_loss = loss_data['train_loss'].flatten()
val_loss = loss_data['val_loss'].flatten()

# Create the figures directory if it doesn't exist
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

# Plot the loss history
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', linewidth=2, color=colors[0])
plt.plot(val_loss, label='Validation Loss', linewidth=2, color=colors[1])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'loss_history.svg'), format='svg')
# plt.close()

# Load the signal data
signal_data = loadmat('prd_test.mat')
actual = signal_data['actual']
denosed = signal_data['denoised']
predicted = signal_data['prediction']

# Select 4 random indices
num_samples = actual.shape[0]
random_indices = random.sample(range(num_samples), 4)

# Plot signal comparisons
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Update linewidth to 1.5 for comparison plots
for i, idx in enumerate(random_indices):
    axes[i].plot(actual[idx, :, 0], label='Actual', linewidth=1.5, color=colors[0])
    axes[i].plot(denosed[idx, :, 0], label='STFT-NMF', linewidth=1.5, color=colors[1])
    axes[i].plot(predicted[idx, :, 0], label='Autoencoder', linewidth=1.5, color=colors[2])
    axes[i].set_title(f'Sample {idx + 1}')

# Calculate RMSE for train and test data
train_rmse = np.sqrt(np.mean((actual - denosed) ** 2))
test_rmse = np.sqrt(np.mean((actual - predicted) ** 2))

# Print RMSE values
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Add a single legend for all subplots
fig.legend(['Actual', 'STFT-NMF', 'Autoencoder'], loc='lower center', ncol=3, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(figures_dir, 'signal_comparison.svg'), format='svg')
# plt.close()

plt.show()