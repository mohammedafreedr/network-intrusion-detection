from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    error = None

    if request.method == "POST":
        try:
            file = request.files["file"]

            if file.filename == "":
                error = "Please select a CSV file."
            else:
                data = pd.read_csv(file)
                data.columns = data.columns.str.strip()

                target = data.columns[-1]

                y = data[target].apply(
                    lambda x: 0 if str(x).strip().lower() == "normal" else 1
                )

                X = data.drop(columns=[target])
                X = pd.get_dummies(X)

                X = VarianceThreshold(0).fit_transform(X)

                k = min(20, X.shape[1])
                X = SelectKBest(f_classif, k=k).fit_transform(X, y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                svm = SVC(kernel="rbf", random_state=42)
                svm.fit(X_train, y_train)
                svm_pred = svm.predict(X_test)
                svm_accuracy = accuracy_score(y_test, svm_pred) * 100

                ann = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=300,
                    random_state=42
                )
                ann.fit(X_train, y_train)
                ann_pred = ann.predict(X_test)
                ann_accuracy = accuracy_score(y_test, ann_pred) * 100

                svm_report = classification_report(
                    y_test, svm_pred,
                    target_names=["Normal", "Attack"],
                    output_dict=True
                )

                ann_report = classification_report(
                    y_test, ann_pred,
                    target_names=["Normal", "Attack"],
                    output_dict=True
                )

                svm_cm = confusion_matrix(y_test, svm_pred)
                ann_cm = confusion_matrix(y_test, ann_pred)

                normal_count = int((y == 0).sum())
                attack_count = int((y == 1).sum())

                if svm_accuracy >= ann_accuracy:
                    best_model = "SVM"
                    best_accuracy = svm_accuracy
                else:
                    best_model = "ANN"
                    best_accuracy = ann_accuracy

                results = {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "target": target,
                    "features": X.shape[1],
                    "svm_accuracy": round(svm_accuracy, 2),
                    "ann_accuracy": round(ann_accuracy, 2),
                    "best_model": best_model,
                    "best_accuracy": round(best_accuracy, 2),
                    "normal_count": normal_count,
                    "attack_count": attack_count,
                    "svm_report": svm_report,
                    "ann_report": ann_report,
                    "svm_cm": svm_cm.tolist(),
                    "ann_cm": ann_cm.tolist()
                }

        except Exception as e:
            error = str(e)

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))