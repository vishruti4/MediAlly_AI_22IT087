from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import os
import random
import logging
import smtplib
from dotenv import load_dotenv
from datetime import datetime, timedelta
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from triage import get_icd10_code, triage_category
from model import symptom_model, get_bert_embedding, label_encoder

# ✅ Load environment variables
load_dotenv(override=True)

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# ✅ MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
mongo_db = client["medially"]
users_collection = mongo_db["users"]
otps_collection = mongo_db["otps"]

# ✅ Load BioBERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)

@app.route("/predict", methods=["POST"])
def predict():
    if symptom_model is None:
        return jsonify({"error": "Model is not loaded"}), 500
    if label_encoder is None:
        return jsonify({"error": "Label encoder is not loaded"}), 500

    data = request.json
    text = data.get("symptoms", "").strip()

    if not text:
        return jsonify({"error": "No symptoms provided"}), 400

    embedding = get_bert_embedding([text])

    with torch.no_grad():
        output = symptom_model(embedding)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_classes = torch.topk(probabilities, 3)

    predictions = []
    for i in range(3):
        disease = label_encoder.inverse_transform([top_classes[0][i].item()])[0]
        confidence = top_probs[0][i].item()
        icd_code = get_icd10_code(disease)
        predictions.append({
            "disease": disease,
            "confidence": confidence,
            "icd_code": icd_code
        })

    # Use the top prediction to determine triage
    main_prediction = predictions[0]
    triage = triage_category(main_prediction["disease"], main_prediction["confidence"])

    return jsonify({
        "predicted_disease": main_prediction["disease"],
        "confidence": float(main_prediction["confidence"]),
        "triage_category": triage,
        "top_predictions": predictions
    })


# ✅ Email Configuration
def send_email_otp(email, otp):
    try:
        email_user = os.getenv("EMAIL_USER")
        email_pass = os.getenv("EMAIL_PASS")
        if not email_user or not email_pass:
            raise ValueError("❌ Email credentials not found!")

        msg = MIMEMultipart()
        msg["From"] = email_user
        msg["To"] = email
        msg["Subject"] = "🔐 Your MediAlly OTP"
        msg.attach(MIMEText(f"Your OTP is: {otp}\nExpires in 5 minutes.", "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_user, email_pass)
        server.sendmail(email_user, email, msg.as_string())
        server.quit()

        logging.info("✅ Email sent successfully!")
        return True
    except Exception as e:
        logging.error(f"❌ Email sending failed: {e}")
        return False

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    # Check if user already exists
    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists."}), 409
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered."}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    })

    return jsonify({"message": "User registered successfully!"}), 200


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    user = users_collection.find_one({"username": username})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    otp = str(random.randint(100000, 999999))
    expiry_time = datetime.utcnow() + timedelta(minutes=5)
    if not send_email_otp(user["email"], otp):
        return jsonify({"error": "Failed to send OTP."}), 500

    otps_collection.insert_one({
        "username": username,
        "otp_email": otp,
        "expires_at": expiry_time,
        "purpose": "login"
    })
    return jsonify({"message": "OTP sent to email."}), 200

@app.route("/verify_login_otp", methods=["POST"])
def verify_login_otp():
    data = request.json
    username = data.get("username")
    otp_email = data.get("otp_email")
    otp_record = otps_collection.find_one({"username": username, "purpose": "login"}, sort=[("expires_at", -1)])
    if not otp_record or str(otp_record["otp_email"]) != str(otp_email):
        return jsonify({"error": "Invalid OTP"}), 401
    if datetime.utcnow() > otp_record["expires_at"]:
        return jsonify({"error": "OTP expired. Request a new one."}), 403
    otps_collection.delete_one({"username": username, "purpose": "login"})
    return jsonify({"message": "OTP verified. Login successful!"}), 200

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


