# 📧 Spam Email Detection

A machine learning web app that detects spam emails using a **Naive Bayes classifier** and **Claude AI**. Built with Python and Flask.

---

## 🚀 Features

- ⚡ Hybrid detection — ML model + Claude AI working together
- 🤖 3 modes: ML Only, ML + AI, AI Only
- 📊 Confidence score for every prediction
- 💡 AI explains *why* a message is spam
- 📱 Works on mobile and desktop
- 🕓 Recent check history + live stats

---

## 🖼️ Demo

> Type any message → get instant spam/ham result with confidence score and reason.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML Model | Naive Bayes (scikit-learn) |
| AI | Claude API (Anthropic) |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Dataset | SMS Spam Collection (5572 messages) |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-detection.git
cd spam-email-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install flask scikit-learn pandas requests
```

### 4. Add your Claude API key
Get a free API key from [console.anthropic.com](https://console.anthropic.com)

```bash
export ANTHROPIC_API_KEY="your-api-key-here"   # Mac/Linux
set ANTHROPIC_API_KEY=your-api-key-here         # Windows
```

### 5. Run the app
```bash
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:8080
```

---

## 📁 Project Structure

```
spam-email-detection/
├── app.py              # Flask backend + ML model + AI integration
├── mail_data.csv       # SMS Spam Collection dataset
├── templates/
│   └── index.html      # Frontend website
└── README.md
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 96% |
| Precision | 100% |
| Recall | 72% |
| F1 Score | 84% |

---

## 🧠 How It Works

1. Message is cleaned (lowercase, remove punctuation)
2. Converted to numbers using TF-IDF
3. Naive Bayes model gives a first prediction
4. Claude AI analyzes the message independently
5. If both agree → confidence is averaged
6. If they disagree → Claude AI result is used

---

## 👨‍💻 Author

**Kaushik**  
Internship Project — Python & AI/ML
