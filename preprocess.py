import pandas as pd
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # Label encode categorical variables (excluding Claim_Description)
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Claim_Description':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    joblib.dump(label_encoders, 'encoder.pkl')

    # BERT embeddings for text
    df['Claim_Description_Embedding'] = df['Claim_Description'].apply(get_bert_embedding)

    # Prepare final features
    text_embeddings = np.vstack(df['Claim_Description_Embedding'].values)
    df.drop(['Claim_ID', 'Claim_Description', 'Claim_Description_Embedding'], axis=1, inplace=True)

    X = np.hstack([df.drop('Fraud_Label', axis=1).values, text_embeddings])
    y = df['Fraud_Label'].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    return X_scaled, y
