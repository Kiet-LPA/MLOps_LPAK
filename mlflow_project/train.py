import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

mlflow.set_tracking_uri("file:./mlruns")  # Lưu log cục bộ
mlflow.set_experiment("lpak_classification")

def train_and_log_model(n_estimators, max_depth):
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "lpak_model")

        print(f"n_estimators={n_estimators}, max_depth={max_depth}, acc={acc:.4f}, f1={f1:.4f}")
        return acc, f1, mlflow.active_run().info.run_id


if __name__ == "__main__":
    results = []
    # Thử nghiệm 3 lần (tuning)
    for n, d in [(50, 3), (100, 5), (150, 7)]:
        acc, f1, run_id = train_and_log_model(n, d)
        results.append((acc, f1, run_id))

    # Chọn model tốt nhất
    best = max(results, key=lambda x: x[0])
    best_run = best[2]

    # Đăng ký model tốt nhất vào Registry
    mlflow.register_model(
        f"runs:/{best_run}/lpak_model",
        "lpak_classifier"
    )

    print(f"✅ Best model logged & registered from run {best_run}")
