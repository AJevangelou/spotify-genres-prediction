# ----- Define IMPORTS ------
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import torch
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import Isomap, TSNE, MDS
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from torch import nn, optim

data = pd.read_csv('/Users/alex/PycharmProjects/spotifyGenresPrediction/spotiy.csv')
print(data.iloc[:, :20])

print(data.nunique())

# Drop unnecessary data
df = data.drop(["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "time_signature", "explicit"], axis=1)

print(df.describe())

print(df.info())

print(df["track_genre"].value_counts())

ax = sns.histplot(df["track_genre"])
_ = plt.xticks(rotation=60)
_ = plt.title("Genres")
plt.show()

X = df.loc[:, :"tempo"]
y = df["track_genre"]

k = 0
plt.figure(figsize=(14, 12))
for i in X.columns:
    plt.subplot(4, 4, k + 1)
    sns.distplot(X[i])
    plt.xlabel(i, fontsize=11)
    k += 1
plt.show()

# -------MODEL TRAINING------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(X_train.columns)

# Standardize features
scalerx = MinMaxScaler()
X_train = scalerx.fit_transform(X_train)
X_test = scalerx.fit_transform(X_test)
xtrain = pd.DataFrame(X_train, columns=X.columns)
xtest = pd.DataFrame(X_test, columns=X.columns)

# Encode labels
le = LabelEncoder()
ytrain = le.fit_transform(y_train)
ytest = le.transform(y_test)

# Combine data for later use
x = pd.concat([xtrain, xtest], axis=0)
y = pd.concat([pd.DataFrame(ytrain), pd.DataFrame(ytest)], axis=0)

# Visualize correlations
plt.subplots(figsize=(8, 6))
ax = sns.heatmap(xtrain.corr()).set(title="Correlation between Features")
plt.show()

# PCA Transform
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x, y)
plot_pca = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
unique_labels = np.unique(y)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i), markersize=10) for i in range(len(unique_labels))]
lg = plt.legend(handles, list(np.unique(y)), loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA")
plt.show()

x = df.loc[:,:"tempo"]
y = df["track_genre"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.2,
                                       random_state=42, shuffle = True)

col = xtrain.columns
scalerx = MinMaxScaler()

xtrain = scalerx.fit_transform(xtrain)
xtest = scalerx.transform(xtest)

xtrain = pd.DataFrame(xtrain, columns = col)
xtest = pd.DataFrame(xtest, columns = col)
le = preprocessing.LabelEncoder()
ytrain = le.fit_transform(ytrain)
ytest = le.transform(ytest)

x = pd.concat([xtrain, xtest], axis = 0)
y = pd.concat([pd.DataFrame(ytrain), pd.DataFrame(ytest)], axis = 0)

y_train = le.inverse_transform(ytrain)
y_test = le.inverse_transform(ytest)
y_org = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis = 0)

# Convert labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
ytrain_encoded = label_encoder.fit_transform(ytrain)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(ytrain_encoded)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_tensor = torch.FloatTensor(X_train_scaled)

# Define the PyTorch model
class CustomModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instantiate the model
input_size = X_train_tensor.shape[1]
num_classes = len(np.unique(y_train))
print(f"Num classes: {num_classes}")
model = CustomModel(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.04)

# Convert to GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

# Training loop
epochs = 250
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "pytorch_model.pth")

model.eval()  # Set the model to evaluation mode
# Convert xtest and ytest to PyTorch tensors
# Convert xtest to a PyTorch tensor
X_test_tensor = torch.FloatTensor(xtest.values)
X_test_tensor = X_test_tensor.to(device)
# Disable MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""

y_test_tensor = torch.LongTensor(label_encoder.transform(ytest))
y_test_tensor = y_test_tensor.to(device)


with torch.no_grad():
    model.to(device)
    X_test_tensor.to(device)
    # Forward pass on the test set
    outputs_test = model(X_test_tensor)

    # Calculate the loss using the same criterion used during training
    loss_test = criterion(outputs_test, y_test_tensor)
    print(f"Test Loss: {loss_test.item()}")

    # Calculate accuracy
    _, predicted_labels = torch.max(outputs_test, 1)
    correct_predictions = (predicted_labels == y_test_tensor).sum().item()
    total_samples = y_test_tensor.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")