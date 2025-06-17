from flask import Blueprint, render_template, request
from app.predict import predict_image

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file:
            result, confidence = predict_image(file)
    return render_template("index.html", result=result, confidence=confidence)
