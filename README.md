# 🧠 BERT Fake Review Detector

This project uses a fine-tuned `DistilBERT` transformer model to classify product reviews as either:
- **CG** – Computer-Generated (AI-written)
- **OR** – Original (Human-written)

Built with PyTorch, HuggingFace Transformers, and trained in Google Colab, this model achieves **98% accuracy** on a labeled dataset of online reviews.

---

## 📊 Dataset

The dataset contains product reviews labeled as either `CG` or `OR`, along with review text, rating, and category metadata.

| Field        | Description                                |
|--------------|--------------------------------------------|
| `text_`      | Review text                                |
| `label`      | Target label: `CG` (1) or `OR` (0)         |
| `rating`     | Star rating (1–5)                          |
| `category`   | Product category (Home, Kitchen, etc.)     |

📦 **Download the dataset** here:  
[Fake vs Real Reviews CSV ](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

---

## 🛠️ Tech Stack

- **Model**: `DistilBERT` (`distilbert-base-uncased`)
- **Framework**: `PyTorch` + `HuggingFace Transformers`
- **Tokenization**: `AutoTokenizer`
- **Training**: Google Colab + GPU (T4)
- **Evaluation**: `sklearn.metrics.classification_report`
- **Deployment Ready**: Model saved in HuggingFace format for use with FastAPI

---

## 🧪 Results

          precision    recall  f1-score   support

      CG       0.97      0.99      0.98      4016
      OR       0.99      0.97      0.98      4071

accuracy                           0.98      8087

---

## 🚀 Usage

### 🔍 Local Inference (Example)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("bert_review_model")
tokenizer = AutoTokenizer.from_pretrained("bert_review_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return "CG" if pred == 1 else "OR"

print(predict("This product is amazing! Highly recommend it."))
