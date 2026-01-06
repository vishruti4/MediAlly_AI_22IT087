import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import joblib
import requests  # ✅ Import requests


LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")

if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Label encoder loaded successfully!")
else:
    print("❌ label_encoder.pkl not found! Please check the file path.")
    label_encoder = None

# ✅ Define SymptomClassifier BEFORE loading the model
class SymptomClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SymptomClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ✅ Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load BioBERT tokenizer and model
# Disable SSL verification globally (use only if necessary)
os.environ["CURL_CA_BUNDLE"] = ""

# Fix SSL issue for Hugging Face requests
requests.packages.urllib3.disable_warnings()

tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    use_auth_token=False,
    trust_remote_code=True  # Optional: trusts remote code execution
)
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)

# ✅ Define model path
MODEL_PATH = os.getenv("MODEL_PATH", "/app/medially.pth")
num_classes = 24  # Adjust based on dataset

# ✅ Initialize symptom model
symptom_model = SymptomClassifier(input_size=768, num_classes=num_classes).to(device)

if os.path.exists(MODEL_PATH):
    symptom_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # ✅ Load weights correctly
    symptom_model.eval()
    print("✅ Symptom Classification Model Loaded!")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")
    symptom_model = None  # Model remains None if not found

def get_bert_embedding(text_list):
    tokens = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = biobert_model(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token (768)
    return embedding  # Shape: [batch_size, 768]


# ✅ Define a function to make predictions
def predict_disease(text, label_encoder):
    if symptom_model is None:
        raise ValueError("❌ Model is not loaded. Cannot make predictions.")

    # Ensure `get_bert_embedding` is defined somewhere
    embedding = get_bert_embedding([text])  # Shape: [1, 768]

    if embedding.shape[1] != 768:
        raise ValueError(f"Embedding shape mismatch! Expected [1, 768], got {embedding.shape}")

    with torch.no_grad():
       output = symptom_model(embedding)  # Ensure model expects [1, 768]

    probabilities = F.softmax(output, dim=1)

    top_prob, top_class = torch.topk(probabilities, 1)
    disease = label_encoder.inverse_transform([top_class[0].item()])[0]
    confidence = top_prob[0].item()


    return disease, confidence
