# ---------------------------
# Stage 1: Build Flask + MLflow App
# ---------------------------

# Base image Python nhẹ, ổn định
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Sao chép thư mục mlruns (chứa MLflow models) vào container
COPY mlruns ./mlruns

# Sao chép script sửa đường dẫn Windows
COPY fix_mlflow_paths.py ./

# Sửa đường dẫn Windows tuyệt đối trong metadata thành đường dẫn Linux
RUN python fix_mlflow_paths.py /app/mlruns

# Thiết lập biến môi trường MLflow tracking URI
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Expose cổng Flask
EXPOSE 5000

# Lệnh khởi chạy Flask app
CMD ["python", "flask_app/app.py"]
