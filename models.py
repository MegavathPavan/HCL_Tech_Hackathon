import torch
import torch.nn as nn
from transformers import BertModel

# Transformer-based Claim Extractor
class ClaimExtractor(nn.Module):
    def __init__(self):
        super(ClaimExtractor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)  # Example: Amount, Injury type, Incident type

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.fc(cls_output)

# Autoencoder for anomaly detection
class FraudAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(FraudAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
