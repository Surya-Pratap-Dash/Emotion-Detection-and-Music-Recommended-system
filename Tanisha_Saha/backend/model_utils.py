import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

id2label = {0: "sad", 1: "happy", 2: "angry", 3: "calm"}

def load_model_and_tokenizer(model_path, tokenizer_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer, device

def predict_text(model, tokenizer, device, text, max_length=96):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).cpu().item()
    return id2label[pred]
