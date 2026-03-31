"""
Gradient Descent for Finding Minimum Error in AI Models
Vector Calculus Practical – 4th Semester
Topic 1: Gradient Descent for Minimizing Prediction Error (House Price Prediction)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. DATASET  (Synthetic House Price Data)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 50
X_raw = np.random.uniform(500, 3000, n)          # Area in sq ft
y = 0.15 * X_raw + 20 + np.random.normal(0, 10, n)  # Price in lakhs

# Normalise feature to [0, 1] for stable gradient descent
X = (X_raw - X_raw.mean()) / X_raw.std()

# ─────────────────────────────────────────────
# 2. MATHEMATICAL MODEL
# ─────────────────────────────────────────────
# Hypothesis:  ŷ = w*x + b
# Loss (MSE):  L = (1/2n) * Σ(ŷᵢ - yᵢ)²
#
# Gradient (vector calculus):
#   ∂L/∂w = (1/n) * Σ(ŷᵢ - yᵢ) * xᵢ
#   ∂L/∂b = (1/n) * Σ(ŷᵢ - yᵢ)
#
# Weight update rule:
#   w ← w − α * ∂L/∂w
#   b ← b − α * ∂L/∂b

def predict(X, w, b):
    return w * X + b

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2) / 2

def gradients(X, y_true, w, b):
    n = len(X)
    y_pred = predict(X, w, b)
    error  = y_pred - y_true
    dw = np.dot(error, X) / n     # ∂L/∂w  (dot product = vector calculus)
    db = np.sum(error) / n        # ∂L/∂b
    return dw, db

# ─────────────────────────────────────────────
# 3. GRADIENT DESCENT ALGORITHM
# ─────────────────────────────────────────────
def gradient_descent(X, y, lr=0.05, epochs=200):
    w, b = 0.0, 0.0           # Initialise weights
    loss_history = []
    w_history    = []
    b_history    = []

    for epoch in range(epochs):
        dw, db = gradients(X, y, w, b)
        w -= lr * dw           # Update rule: w ← w − α·∇w
        b -= lr * db           # Update rule: b ← b − α·∇b
        loss = mse_loss(predict(X, w, b), y)
        loss_history.append(loss)
        w_history.append(w)
        b_history.append(b)

    return w, b, loss_history, w_history, b_history

# Run
lr = 0.05
epochs = 200
w_final, b_final, loss_history, w_hist, b_hist = gradient_descent(X, y, lr, epochs)

y_pred_final = predict(X, w_final, b_final)
initial_loss = mse_loss(predict(X, 0, 0), y)
final_loss   = loss_history[-1]

print(f"Initial Loss  : {initial_loss:.4f}")
print(f"Final Loss    : {final_loss:.4f}")
print(f"Reduction     : {(1 - final_loss/initial_loss)*100:.2f}%")
print(f"Final weights : w = {w_final:.4f}, b = {b_final:.4f}")

# ─────────────────────────────────────────────
# 4. VISUALISATIONS  (4 plots saved separately)
# ─────────────────────────────────────────────

# --- Plot 1 : Error vs Iterations ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(epochs), loss_history, color='steelblue', linewidth=2)
ax.set_xlabel("Iteration (Epoch)", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_title("Gradient Descent: Error vs Iterations", fontsize=13, fontweight='bold')
ax.fill_between(range(epochs), loss_history, alpha=0.15, color='steelblue')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/claude/plot1_error_vs_iterations.png', dpi=150)
plt.close()
print("Saved: plot1_error_vs_iterations.png")

# --- Plot 2 : Regression line (prediction) ---
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
plt.savefig('/home/claude/plot2_regression_line.png', dpi=150)
plt.close()
print("Saved: plot2_regression_line.png")

# --- Plot 3 : Weight convergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(w_hist, color='green', linewidth=2)
ax1.set_title("Weight (w) Convergence", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("w value")
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(b_hist, color='purple', linewidth=2)
ax2.set_title("Bias (b) Convergence", fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("b value")
ax2.grid(True, linestyle='--', alpha=0.5)
plt.suptitle("Parameter Convergence Over Epochs", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/plot3_weight_convergence.png', dpi=150)
plt.close()
print("Saved: plot3_weight_convergence.png")

# --- Plot 4 : 3D Loss Surface (w vs b vs Loss) ---
from mpl_toolkits.mplot3d import Axes3D
w_range = np.linspace(w_final - 40, w_final + 40, 60)
b_range = np.linspace(b_final - 20, b_final + 20, 60)
W, B = np.meshgrid(w_range, b_range)
L = np.array([[mse_loss(predict(X, wi, bi), y) for wi in w_range] for bi in b_range])

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, L, cmap='viridis', alpha=0.8)
ax.scatter([w_final], [b_final], [final_loss], color='red', s=80, zorder=5, label='Minimum')
ax.set_xlabel("Weight (w)"); ax.set_ylabel("Bias (b)"); ax.set_zlabel("Loss")
ax.set_title("3D Loss Surface (Gradient Landscape)", fontweight='bold')
fig.colorbar(surf, shrink=0.5)
plt.tight_layout()
plt.savefig('/home/claude/plot4_loss_surface.png', dpi=150)
plt.close()
print("Saved: plot4_loss_surface.png")

print("\nAll plots saved successfully.")
print(f"\n=== RESULTS SUMMARY ===")
print(f"Dataset        : {n} synthetic house price records")
print(f"Learning Rate  : {lr}")
print(f"Epochs         : {epochs}")
print(f"Initial Loss   : {initial_loss:.4f}")
print(f"Final Loss     : {final_loss:.4f}")
print(f"Loss Reduction : {(1 - final_loss/initial_loss)*100:.2f}%")
print(f"Final w        : {w_final:.4f}")
print(f"Final b        : {b_final:.4f}")
