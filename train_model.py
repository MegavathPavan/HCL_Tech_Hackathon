import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess_data
from models import FraudAutoEncoder
import joblib

# Prepare data
X, y = preprocess_data('D:/HCLTech/predata/corr_data.csv')

# DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = FraudAutoEncoder(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

# Save model
torch.save(model.state_dict(), 'fraud_autoencoder.pth')
print("Model training complete and saved.")
