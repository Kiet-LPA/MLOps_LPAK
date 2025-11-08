from flask import Flask, render_template, request
import mlflow.pyfunc
import numpy as np

# Tên model trong MLflow Registry
MODEL_URI = "models:/lpak_classifier/1"   # hoặc "models:/lpak_classifier/Production" nếu bạn đặt stage

app = Flask(__name__)

# Load model từ Registry
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    print(f"⚠️ Lỗi khi load model: {e}")
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Đọc 10 feature đầu vào
            features = [float(request.form[f"f{i}"]) for i in range(1, 11)]
            arr = np.array(features).reshape(1, -1)
            prediction = int(model.predict(arr)[0])
        except Exception as e:
            prediction = f"Lỗi khi dự đoán: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
