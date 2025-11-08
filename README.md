# 🧠 MLOps Project – LPak Classifier  
**Tác giả:** Lưu Phạm Anh Kiệt  
**Trường:** FSB – Master of Software Engineering  
**Môn học:** MLOps  
**Deadline:** 17/11/2025  

---

## 🎯 Mục tiêu
Xây dựng một quy trình **MLOps đầy đủ** bao gồm:
1. Sinh dữ liệu huấn luyện mô phỏng bằng `make_classification`
2. Huấn luyện mô hình phân loại (RandomForest)
3. Ghi log quá trình bằng **MLflow Tracking**
4. Lưu và quản lý mô hình bằng **MLflow Model Registry**
5. Tạo ứng dụng web Flask sử dụng mô hình tốt nhất
6. Đóng gói toàn bộ ứng dụng bằng **Docker**
7. (Bonus) Thiết lập CI/CD tự động build & push image lên Docker Hub

---

## 🧩 Cấu trúc thư mục dự án

MLOps/
│
├── mlflow_project/ # Thư mục huấn luyện mô hình
│ ├── train.py # Huấn luyện + log + đăng ký model
│ ├── tuning.py # Thử nghiệm tham số
│ ├── data_generator.py # Sinh dữ liệu mô phỏng
│ └── init.py
│
├── flask_app/ # Ứng dụng Flask load model tốt nhất
│ ├── app.py
│ ├── init.py
│ └── templates/
│ └── index.html
│
├── docker/ # Tài nguyên Docker (nếu có)
│ └── README.md
│
├── .gitlab-ci.yml # CI/CD pipeline (cho bonus 2 điểm)
├── Dockerfile # Đóng gói Flask app
├── requirements.txt # Các gói cần thiết
└── README.md # Hướng dẫn thực hiện (file hiện tại)


---

Bước 1: Cài đặt môi trường
    
    ### Tạo môi trường ảo
    ```bash
    python -m venv venv
    venv\Scripts\activate

    
    Nếu PowerShell báo lỗi "running scripts is disabled", chạy:

        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
        
    Cài đặt thư viện:
        pip install mlflow scikit-learn flask numpy pandas gunicorn matplotlib

Bước 2: Huấn luyện mô hình và ghi log bằng MLflow
File: mlflow_project/train.py

Sử dụng make_classification để sinh dữ liệu mẫu

Huấn luyện mô hình RandomForestClassifier

Ghi log tham số, metric (accuracy, f1_score)

Thực hiện 3 lần tuning

Chọn mô hình có accuracy cao nhất và đăng ký vào MLflow Model Registry

Chạy:

python mlflow_project/train.py
Kết quả mẫu:

n_estimators=50, max_depth=3, acc=0.8600, f1=0.8704
n_estimators=100, max_depth=5, acc=0.8650, f1=0.8744
n_estimators=150, max_depth=7, acc=0.8750, f1=0.8848
✅ Best model logged & registered from run ...

Kiểm tra MLflow UI:

    mlflow ui
    
    
    → Truy cập http://127.0.0.1:5000
    
Bước 3: Tạo ứng dụng Flask để dự đoán bằng model tốt nhất
    
    File: flask_app/app.py
    
    Load mô hình từ mlflow.pyfunc.load_model("models:/lpak_classifier/1")
    
    Nhận dữ liệu đầu vào gồm 10 feature
    
    Trả về kết quả phân loại (0 hoặc 1)
    
    Giao diện web (flask_app/templates/index.html) có form nhập liệu đơn giản:
    
    <h2>Dự đoán kết quả phân loại (LPak Classifier)</h2>
    <form method="POST">
      f1–f10: nhập giá trị số
    </form>
    
    
    Chạy:
    
    python flask_app/app.py
    
    
    Truy cập http://127.0.0.1:5000
    
     Nếu thấy giao diện như hình dưới và dự đoán ra 0 hoặc 1 → mô hình Flask đã hoạt động thành công.
    
Bước 4: Đóng gói ứng dụng bằng Docker
    
    File: Dockerfile
    
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 5000
    CMD ["python", "flask_app/app.py"]
    
    
    Build image:
    
    docker build -t lpak-mlops .
    
    
    Chạy container:
    
    docker run -p 5000:5000 lpak-mlops
    
    
    Truy cập http://127.0.0.1:5000
    
    → Giao diện Flask vẫn chạy bình thường trong container.

---

## 🐳 **Bước 5: Build & Chạy Docker Container cho Flask App**

Sau khi hoàn thiện ứng dụng Flask, ta tiến hành đóng gói vào **Docker container** để có thể chạy ở bất kỳ môi trường nào mà không cần cài đặt Python hay MLflow.

### 🧱 1. Cấu hình Dockerfile

File: `Dockerfile`
```dockerfile
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

# Mở cổng Flask
EXPOSE 5000

# Lệnh khởi chạy Flask app
CMD ["python", "flask_app/app.py"]

2. Build Docker image

Dừng Flask app đang chạy (nếu có):

Ctrl + C


Sau đó build image mới:

docker build -t lpak-mlops .


Kết quả khi thành công:

Successfully built <container_id>
Successfully tagged lpak-mlops:latest

🧩 3. Kiểm tra image
docker images


Output mẫu:

REPOSITORY     TAG       IMAGE ID       CREATED          SIZE
lpak-mlops     latest    3b1b2cd45678   1 minute ago     700MB

▶️ 4. Chạy container Flask
docker run -p 5000:5000 lpak-mlops


Truy cập trình duyệt:
👉 http://127.0.0.1:5000

Ứng dụng Flask hiển thị form dự đoán tương tự bản local trước đó, chứng minh rằng toàn bộ hệ thống MLflow + Flask đã chạy được trong môi trường container hóa.