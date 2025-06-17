# MNIST Digit Classifier (Flask + PyTorch + Web)

Upload a hand-written digit image (0-9) or draw it, and get prediction + confidence.

## 🔧 Setup

```bash
git clone <your-repo-url>
cd mnist-digit-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train/train_model.py
python app.py
```

Visit http://127.0.0.1:5000 to use the app.

## 🐳 Docker (optional)

```bash
docker build -t mnist-app .
docker run -p 5000:5000 mnist-app
```

## ✅ Features
- PyTorch-trained digit classifier
- Flask web app with file upload + drawing canvas
- Logging of inferences
- Ready for cloud deployment