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

# Expose cổng Flask
EXPOSE 5000

# Lệnh khởi chạy Flask app
CMD ["python", "flask_app/app.py"]
