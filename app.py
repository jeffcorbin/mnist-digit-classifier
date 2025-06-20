from flask import Flask
from app.routes import main
import os

app = Flask(__name__)
app.register_blueprint(main)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
