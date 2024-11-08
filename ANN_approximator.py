import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

sns.set()
import torch
import torch.optim as optim
import torch.nn as nn
from IPython import display
from scipy.stats import multivariate_normal


mean = [0, 0]
cov = [[1, 1], [1, 4]]
var = multivariate_normal(mean=mean, cov=cov)

X_train = np.random.multivariate_normal(mean, cov, 1000)
y_train = var.pdf(X_train)

# Plot the training data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = var.pdf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

ax.plot_surface(X, Y, Z, color="blue", alpha=0.5)

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("PDF")
plt.show()

plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.reshape(len(X_train), -1)
x_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

model = nn.Sequential(
    nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
)
model = model.to(device)

# Linear Model
learning_rate = 0.01  # alpha
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_epoch = 5000
Losses = []
# Training
for t in range(n_epoch):
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred.T, y_train_tensor)
    optimizer.zero_grad()
    if t % 100:
        Losses.append(loss.item())
    loss.backward()
    optimizer.step()
plt.plot(Losses)
plt.show()


# Generate a grid of test points in 2D
x1_test = np.linspace(-3, 3, 100)
x2_test = np.linspace(-3, 3, 100)
x1_test, x2_test = np.meshgrid(x1_test, x2_test)
x_test = np.column_stack([x1_test.ravel(), x2_test.ravel()])

# Compute the true PDF values for the test points
y_test = var.pdf(x_test)

# Convert test points to tensor
x_test_tensor = torch.from_numpy(x_test).float().to(device)

# Predict the PDF values using the model
y_pred = model(x_test_tensor)
y_pred = y_pred.flatten().cpu().detach().numpy()

# Reshape the results to match the grid shape
y_test = y_test.reshape(x1_test.shape)
y_pred = y_pred.reshape(x1_test.shape)

# Plot the true PDF values in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1_test, x2_test, y_test, color="blue", alpha=0.5, label="True PDF")

# Plot the predicted PDF values in 3D
ax.plot_surface(x1_test, x2_test, y_pred, color="red", alpha=0.5, label="Predicted PDF")

ax.set_xlabel("X1 axis")
ax.set_ylabel("X2 axis")
ax.set_zlabel("PDF")
plt.legend()
plt.show()

# Compute the MSE and MAE for the entire test set
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error for the entire test set: {mse:.6f}")
print(f"Mean Absolute Error for the entire test set: {mae:.6f}")


# Extract the corresponding x1_test and y_pred values
x1_section = x1_test[0, :]
y_pred_section = y_pred[50, :]
y_test_section = y_test[50, :]

# Plot the section at y = 0
plt.plot(x1_section, y_test_section, ".", label="True PDF at y=0")
plt.plot(x1_section, y_pred_section, ".", label="Predicted PDF at y=0")
plt.xlabel("x1")
plt.ylabel("PDF")
plt.legend()
plt.title("Section at y = 0")
plt.show()

# Plot the contour plots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)

# True PDF contour plot
cp1 = ax1.contourf(X, Y, y_test, cmap="viridis")
fig.colorbar(cp1, ax=ax1)
ax1.set_title("True PDF Contour Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Predicted PDF contour plot
cp2 = ax2.contourf(X, Y, y_pred, cmap="plasma")
fig.colorbar(cp2, ax=ax2)
ax2.set_title("Predicted PDF Contour Plot")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.show()
