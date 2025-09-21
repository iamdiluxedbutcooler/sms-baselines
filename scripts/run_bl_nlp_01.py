cat > /content/drive/MyDrive/sms-baselines/scripts/run_bl_nlp_01.py <<'PY'
import os, json, pandas as pd
from src.models.nlp_baselines import NaiveBayesSmishing, evaluate_cv
train = pd.read_csv("/content/drive/MyDrive/sms-baselines/data/splits/train.csv")
test = pd.read_csv("/content/drive/MyDrive/sms-baselines/data/splits/test.csv")
texts_train = train['text'].tolist()
labels_train = train['label'].tolist()
texts_test = test['text'].tolist()
labels_test = test['label'].tolist()
cv_res = None
try:
    cv_res = evaluate_cv(texts_train, labels_train, ngram_range=(1,2), max_features=20000, alpha=1.0, cv=5)
except Exception:
    cv_res = None
model = NaiveBayesSmishing(ngram_range=(1,2), max_features=20000, alpha=1.0)
model.fit(texts_train, labels_train)
out_dir = "/content/drive/MyDrive/sms-baselines/experiments/BL_NLP_01_nb_full"
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "naive_bayes_smishing.joblib")
model.save(model_path)
preds = model.predict(texts_test)
probs = None
try:
    probs = model.predict_proba(texts_test)
except Exception:
    probs = None
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = float(accuracy_score(labels_test, preds))
report = classification_report(labels_test, preds, output_dict=True)
cm = confusion_matrix(labels_test, preds).tolist()
results = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm, "model_path": model_path, "cv": cv_res}
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
df_out = pd.DataFrame({"text": texts_test, "true_label": labels_test, "pred_label": preds})
if probs is not None:
    for i in range(probs.shape[1]):
        df_out[f"prob_{i}"] = probs[:, i]
df_out.to_csv(os.path.join(out_dir, "results_detailed.csv"), index=False)
print("done. artifacts saved to", out_dir)
PY
