from flask import Flask, render_template, request
import mlflow.pyfunc
import mlflow
import numpy as np
import os

# Thiết lập MLflow tracking URI (cho Docker container)
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"📌 MLflow tracking URI: {mlflow_tracking_uri}")

app = Flask(__name__)

# Load model - sử dụng nhiều phương pháp để đảm bảo hoạt động trong Docker
# Model version 1 được đăng ký từ run_id: 545cfe034e9f4902944e82b745e5e7a7
# experiment_id: 602464189259425114
model = None
# Sử dụng base path từ MLFLOW_TRACKING_URI hoặc mặc định
base_mlruns_path = mlflow_tracking_uri if os.path.isabs(mlflow_tracking_uri) else os.path.join(os.getcwd(), mlflow_tracking_uri)

# Đường dẫn tuyệt đối đến model artifacts trong container
direct_model_path = os.path.join(base_mlruns_path, "602464189259425114/models/m-5c70dc42de7f4e63ad68f8ad473ae8f4/artifacts")

model_paths = [
    # Ưu tiên 1: Load trực tiếp từ artifacts folder (hoạt động sau khi đã fix paths)
    direct_model_path,
    # Ưu tiên 2: Load từ registry (sau khi đã sửa đường dẫn Windows trong Dockerfile)
    "models:/lpak_classifier/1",
    # Ưu tiên 3: Load từ run_id trực tiếp (fallback)
    "runs:/545cfe034e9f4902944e82b745e5e7a7/model",
]

for model_path in model_paths:
    try:
        print(f"🔄 Attempting to load model from: {model_path}")
        # Kiểm tra path tồn tại nếu là đường dẫn file system
        if isinstance(model_path, str) and not model_path.startswith(("models:", "runs:")):
            if not os.path.exists(model_path):
                print(f"⚠️ Path does not exist: {model_path}")
                continue
        
        model = mlflow.pyfunc.load_model(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        break
    except Exception as e:
        print(f"⚠️ Failed to load from {model_path}: {str(e)}")
        continue

if model is None:
    print("❌ Failed to load model from all paths!")
    print("💡 Please ensure mlruns directory is properly copied into Docker container.")
else:
    print("✅ Model is ready for predictions!")

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
