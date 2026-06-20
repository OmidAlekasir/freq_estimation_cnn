import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.io import loadmat
from utils import data_to_tensors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from tensorflow.keras.models import load_model

# --- Data Preparation ---
path = "../../MSC/matlab_codes/patients_dataset"
data = loadmat(path)
X, Y = data_to_tensors(data)
X = np.array(X).astype(np.float32)
Y = np.array(Y)
bpm_max = 180
X /= np.max(np.abs(X))
Y /= bpm_max
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

# --- Load Model ---
model = load_model("CNN_86_new.h5")


# --- Metrics Functions ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_metrics(metrics_dict):
    names = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color=["skyblue", "lightgreen", "salmon"])
    plt.title("Regression Error Metrics")
    plt.ylabel("Error")
    plt.show()


# --- Predictions ---
Y_pred = model.predict(X).flatten()
Y_test_pred = model.predict(X_test).flatten()

# --- Evaluation ---
percentage_test1 = (
    np.sum(np.abs(Y_test * bpm_max - Y_test_pred * bpm_max) < 3) / len(Y_test) * 100
)
percentage_test2 = (
    np.sum(np.abs(Y_test * bpm_max - Y_test_pred * bpm_max) < 6) / len(Y_test) * 100
)
percentage_test3 = (
    np.sum(np.abs(Y_test * bpm_max - Y_test_pred * bpm_max) < 10) / len(Y_test) * 100
)
print("Percentage of test samples with error < 3 bpm:", percentage_test1, "%")
print("Percentage of test samples with error < 6 bpm:", percentage_test2, "%")
print("Percentage of test samples with error < 10 bpm:", percentage_test3, "%")

metrics = {
    "RMSE": rmse(Y * bpm_max, Y_pred * bpm_max),
    "MAE": mean_absolute_error(Y * bpm_max, Y_pred * bpm_max),
    "RMSE (Test)": rmse(Y_test * bpm_max, Y_test_pred * bpm_max),
    "MAE (Test)": mean_absolute_error(Y_test * bpm_max, Y_test_pred * bpm_max),
}
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# --- Plotting Settings ---
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.color"] = "gray"

# --- Load Loss History if available ---
loss, val_loss = None, None
if os.path.exists("plot_data.npz"):
    plot_data = np.load("plot_data.npz")
    loss = np.maximum(plot_data["loss"], 1e-10)
    val_loss = np.maximum(plot_data["val_loss"], 1e-10)

# --- Create figures directory ---
os.makedirs("figures", exist_ok=True)

# --- Plot Loss History ---
if loss is not None and val_loss is not None:
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss, label="Training Loss", color="crimson", linewidth=2)
    plt.semilogy(val_loss, label="Validation Loss", color="royalblue", linewidth=2)
    plt.title("Model Loss Over Epochs (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "figures/loss_history_log.svg", format="svg", dpi=300, bbox_inches="tight"
    )
    plt.show()

# --- Plot All Data: Actual vs Predicted ---
plt.figure(figsize=(10, 6))
plt.plot(
    Y * bpm_max,
    label="Actual BPM",
    linestyle="-",
    color="#1f77b4",
    linewidth=2,
    alpha=0.7,
)
plt.plot(
    Y_pred * bpm_max,
    label="Predicted BPM",
    linestyle="-",
    color="#ff7f0e",
    linewidth=2,
    alpha=0.7,
)
plt.title("Actual vs Predicted BPM (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("BPM")
plt.legend()
plt.grid(True)
plt.savefig(
    "figures/whole_data_comparison.svg", format="svg", dpi=300, bbox_inches="tight"
)
plt.show()

# --- Plot Test Data: Actual vs Predicted ---
plt.figure(figsize=(10, 6))
plt.plot(
    Y_test * bpm_max,
    label="Actual Test Data",
    linestyle="-",
    color="#2ca02c",
    linewidth=2,
    alpha=0.7,
)
plt.plot(
    Y_test_pred * bpm_max,
    label="Predicted Test Data",
    linestyle="-",
    color="#d62728",
    linewidth=2,
    alpha=0.7,
)
plt.title("Actual vs Predicted BPM (Test Data)")
plt.xlabel("Sample Index")
plt.ylabel("BPM")
plt.legend()
plt.grid(True)
plt.savefig(
    "figures/test_data_comparison.svg", format="svg", dpi=300, bbox_inches="tight"
)
plt.show()
