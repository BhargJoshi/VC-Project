"""
Gradient Descent for Finding Minimum Error in AI Models
Vector Calculus Practical - 4th Semester
Topic 1: House Price Prediction via Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ── Works in both Jupyter Notebook AND as a .py script ───────────────────────
try:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SAVE_DIR = os.getcwd()   # Jupyter: saves in the current working directory

print(f"Plots will be saved to: {SAVE_DIR}")

# =============================================================================
# 1. DATASET  (Synthetic House Price Data)
# =============================================================================
np.random.seed(42)
n = 50
X_raw = np.random.uniform(500, 3000, n)                      # Area in sq ft
y     = 0.15 * X_raw + 20 + np.random.normal(0, 10, n)      # Price in lakhs

# Normalise feature to zero mean, unit variance (stable gradient descent)
X_MEAN = X_raw.mean()   # ≈ 1614.81  — used by website for live normalisation
X_STD  = X_raw.std()    # ≈  714.95  — used by website for live normalisation
X = (X_raw - X_MEAN) / X_STD

print(f"Dataset normalisation → mean={X_MEAN:.4f}, std={X_STD:.4f}")

# =============================================================================
# 2. CORE FUNCTIONS  (Mathematical Model)
# =============================================================================

def predict(X, w, b):
    """Hypothesis: y_hat = w*x + b"""
    return w * X + b

def mse_loss(y_pred, y_true):
    """Loss: L = (1/2n) * sum((y_hat - y)^2)"""
    return np.mean((y_pred - y_true) ** 2) / 2

def gradients(X, y_true, w, b):
    """
    Gradient of loss w.r.t. weights (vector calculus):
      dL/dw = (1/n) * dot(error, X)
      dL/db = (1/n) * sum(error)
    """
    n_     = len(X)
    y_pred = predict(X, w, b)
    error  = y_pred - y_true
    dw     = np.dot(error, X) / n_
    db     = np.sum(error) / n_
    return dw, db

# =============================================================================
# 3. GRADIENT DESCENT ALGORITHM
# =============================================================================

def gradient_descent(X, y, lr=0.05, epochs=200):
    w, b         = 0.0, 0.0
    loss_history = []
    w_history    = []
    b_history    = []

    for epoch in range(epochs):
        dw, db = gradients(X, y, w, b)
        w -= lr * dw          # w <- w - alpha * dL/dw
        b -= lr * db          # b <- b - alpha * dL/db
        loss_history.append(mse_loss(predict(X, w, b), y))
        w_history.append(w)
        b_history.append(b)

    return w, b, loss_history, w_history, b_history

# Run
lr, epochs = 0.05, 200
w_final, b_final, loss_history, w_hist, b_hist = gradient_descent(X, y, lr, epochs)

y_pred_final = predict(X, w_final, b_final)
initial_loss = mse_loss(predict(X, 0, 0), y)
final_loss   = loss_history[-1]

# R² score
ss_res = np.sum((y - y_pred_final) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2     = 1 - ss_res / ss_tot

# =============================================================================
# 4. PRINT RESULTS
# =============================================================================
print("=" * 50)
print("         GRADIENT DESCENT - RESULTS")
print("=" * 50)
print(f"  Dataset size        : {n} records")
print(f"  Learning Rate       : {lr}")
print(f"  Epochs              : {epochs}")
print(f"  Initial Loss (MSE)  : {initial_loss:.4f}")
print(f"  Final Loss   (MSE)  : {final_loss:.4f}")
print(f"  Loss Reduction      : {(1 - final_loss/initial_loss)*100:.2f}%")
print(f"  Final w             : {w_final:.4f}")
print(f"  Final b             : {b_final:.4f}")
print(f"  R² Score            : {r2:.4f}  ({r2*100:.1f}%)")
print("=" * 50)
print()
print("  ► These values are used verbatim in index.html")
print(f"    normalisedArea(x) = (x - {X_MEAN:.2f}) / {X_STD:.2f}")
print(f"    ŷ = {w_final:.4f} · x_norm + {b_final:.4f}")
print("=" * 50)

# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

# --- Plot 1 : Error vs Iterations ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(epochs), loss_history, color='steelblue', linewidth=2)
ax.fill_between(range(epochs), loss_history, alpha=0.15, color='steelblue')
ax.set_xlabel("Iteration (Epoch)", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_title("Gradient Descent: Error vs Iterations", fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
path1 = os.path.join(SAVE_DIR, "plot1_error_vs_iterations.png")
plt.savefig(path1, dpi=150)
plt.show()
print(f"Saved: {path1}")

# --- Plot 2 : Regression Line ---
fig, ax = plt.subplots(figsize=(7, 4))
sort_idx = np.argsort(X)
ax.scatter(X, y, color='coral', alpha=0.7, label='Actual data', s=40)
ax.plot(X[sort_idx], y_pred_final[sort_idx], color='navy', linewidth=2, label='Predicted line')
ax.set_xlabel("Normalised Area (x)", fontsize=12)
ax.set_ylabel("House Price (Lakhs)", fontsize=12)
ax.set_title("Linear Regression via Gradient Descent", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
path2 = os.path.join(SAVE_DIR, "plot2_regression_line.png")
plt.savefig(path2, dpi=150)
plt.show()
print(f"Saved: {path2}")

# --- Plot 3 : Weight & Bias Convergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(w_hist, color='green', linewidth=2)
ax1.set_title("Weight (w) Convergence", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("w value")
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(b_hist, color='purple', linewidth=2)
ax2.set_title("Bias (b) Convergence", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("b value")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Parameter Convergence Over Epochs", fontsize=13, fontweight='bold')
plt.tight_layout()
path3 = os.path.join(SAVE_DIR, "plot3_weight_convergence.png")
plt.savefig(path3, dpi=150)
plt.show()
print(f"Saved: {path3}")

# --- Plot 4 : 3D Loss Surface ---
w_range = np.linspace(w_final - 40, w_final + 40, 60)
b_range = np.linspace(b_final - 20, b_final + 20, 60)
W, B    = np.meshgrid(w_range, b_range)
L       = np.array([[mse_loss(predict(X, wi, bi), y)
                     for wi in w_range]
                    for bi in b_range])

fig = plt.figure(figsize=(8, 5))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, L, cmap='viridis', alpha=0.8)
ax.scatter([w_final], [b_final], [final_loss],
           color='red', s=80, zorder=5, label='Minimum')
ax.set_xlabel("Weight (w)")
ax.set_ylabel("Bias (b)")
ax.set_zlabel("Loss")
ax.set_title("3D Loss Surface (Gradient Landscape)", fontweight='bold')
fig.colorbar(surf, shrink=0.5)
plt.tight_layout()
path4 = os.path.join(SAVE_DIR, "plot4_loss_surface.png")
plt.savefig(path4, dpi=150)
plt.show()
print(f"Saved: {path4}")

print("\nAll 4 plots generated and saved successfully.")
