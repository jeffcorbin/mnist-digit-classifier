from flask import Blueprint, render_template, request, jsonify
from app.predict import predict_image, predict_base64_image

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            result, confidence = predict_image(file)
    return render_template("index.html", result=result, confidence=confidence)

@main.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    prediction, confidence = predict_base64_image(data["image"])
    return jsonify({"prediction": prediction, "confidence": confidence})
