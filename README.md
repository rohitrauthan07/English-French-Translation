# English-French-Translation

# English to French Translation using Hugging Face Transformers

This project fine-tunes a pretrained MarianMT (Opus MT) model to translate English to French using a custom dataset loaded directly from Google Sheets.

## 🚀 Features

- Loads translation dataset from a live Google Sheets CSV
- Fine-tunes the `Helsinki-NLP/opus-mt-en-fr` model
- Uses Hugging Face `transformers` and `datasets` libraries
- Trains a model for English → French translation
- Saves the trained model for future inference

---

## 📊 Dataset

The dataset is sourced from this public Google Sheet:  
🔗 [View Dataset](https://docs.google.com/spreadsheets/d/1Ca81JsYuXdwAId6C1y_mqJcAq22LWZtLl8I8_oCBdWU/edit?usp=sharing)

| English        | French          |
|----------------|------------------|
| Hello          | Bonjour          |
| How are you?   | Comment ça va ?  |

Make sure the dataset has two columns labeled `en` and `fr`.

---

## 🛠️ Installation

```bash
pip install pandas transformers datasets sacremoses

## 🛠️ Run
python english_to_french.py
