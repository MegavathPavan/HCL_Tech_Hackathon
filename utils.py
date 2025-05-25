import numpy as np
import torch
import joblib
from models import FraudAutoEncoder
from transformers import BertTokenizer, BertModel

# Load components
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoder.pkl')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

model = FraudAutoEncoder(input_dim=scaler.mean_.shape[0])
model.load_state_dict(torch.load('fraud_autoencoder.pth'))
model.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def preprocess_input(data_dict):
    features = []
    for col, le in label_encoders.items():
        val = data_dict.get(col)
        if val is not None:
            val = le.transform([val])[0]
        else:
            val = 0  # or some default
        features.append(val)

    # Manual numerical fields
    numerical_fields = [
        'Customer_Age', 'Claim_Amount', 'Claim_History', 'Claim_Frequency',
        'Income_Level', 'Coverage_Amount', 'Premium_Amount', 'Deductible'
    ]
    for field in numerical_fields:
        features.append(data_dict.get(field, 0))

    # Text embedding
    text_embedding = get_bert_embedding(data_dict['Claim_Description'])

    # Combine features
    input_data = np.hstack([features, text_embedding])
    input_scaled = scaler.transform([input_data])

    return torch.tensor(input_scaled, dtype=torch.float32)

def predict(data_dict):
    input_tensor = preprocess_input(data_dict)
    with torch.no_grad():
        reconstruction = model(input_tensor)
        loss = torch.mean((input_tensor - reconstruction) ** 2).item()
    threshold = 0.01  # Can be adjusted based on validation
    return 'Fraudulent' if loss > threshold else 'Legitimate'
