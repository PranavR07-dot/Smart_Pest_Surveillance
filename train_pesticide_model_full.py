import pandas as pd
import numpy as np
import joblib
import os
import gc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, mean_squared_error
)
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================
# 1️⃣ Load Dataset
# ==============================================================
print("📥 Loading dataset...")
df = pd.read_csv("Smart_Pesticide_MultiRecommend.csv")

# Simplify multiple pesticides → take only the first one
df["pesticide"] = df["pesticide"].astype(str).apply(lambda x: x.split(",")[0].strip())

# ==============================================================
# 2️⃣ Clean & Prepare Data
# ==============================================================
features = ["Humidity", "Moisture", "Temperature", "Plant Type"]
target = "pesticide"

df = df[features + [target]].dropna().drop_duplicates()
X = df[features]
y = df[target]

print(f"✅ Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
print(f"🌱 Unique pesticide classes: {len(np.unique(y))}")

# ==============================================================
# 3️⃣ Encoding and Pipeline
# ==============================================================
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Plant Type"])
], remainder="passthrough")

# ==============================================================
# 4️⃣ Model Hyperparameters
# ==============================================================
param_dist = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [6, 8, 10, 12],
    'model__min_samples_split': [2, 4],
    'model__min_samples_leaf': [1, 2],
}

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=8,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# ==============================================================
# 5️⃣ Split Data and Train
# ==============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
gc.collect()

print("\n🤖 Training Random Forest models...")
search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"✅ Best Parameters: {search.best_params_}")

# ==============================================================
# 6️⃣ Evaluation (Full Model)
# ==============================================================
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv = cross_val_score(best_model, X, y, cv=5, scoring="accuracy").mean()
mse = mean_squared_error(pd.factorize(y_test)[0], pd.factorize(y_pred)[0])

print("\n🎯 Full Model Performance:")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Cross-Validation: {cv*100:.2f}%")
print(f"MSE: {mse:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================================================
# 7️⃣ Save Full Model (Local Use)
# ==============================================================
os.makedirs("results", exist_ok=True)
joblib.dump(best_model, "pest_prediction_model_full.joblib", compress=1)

# ==============================================================
# 8️⃣ Train Lightweight Model for Streamlit
# ==============================================================
print("\n💡 Training lightweight version for Streamlit Cloud...")

light_rf = RandomForestClassifier(
    n_estimators=80, max_depth=8,
    min_samples_split=3, min_samples_leaf=2,
    random_state=42
)

light_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", light_rf)
])

light_pipe.fit(X_train, y_train)

y_pred_light = light_pipe.predict(X_test)
acc_light = accuracy_score(y_test, y_pred_light)
cv_light = cross_val_score(light_pipe, X, y, cv=3, scoring="accuracy").mean()
mse_light = mean_squared_error(pd.factorize(y_test)[0], pd.factorize(y_pred_light)[0])

print("\n🌿 Lightweight Model Performance:")
print(f"Accuracy: {acc_light*100:.2f}%")
print(f"Cross-Validation: {cv_light*100:.2f}%")
print(f"MSE: {mse_light:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_light))

# Save lightweight version
joblib.dump(light_pipe, "pest_prediction_model_light.joblib", compress=3)
print("✅ Saved lightweight model successfully (under 100 MB).")

# ==============================================================
# 9️⃣ Feature Importance
# ==============================================================
rf_model = best_model.named_steps["model"]
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
ohe_features = list(ohe.get_feature_names_out(["Plant Type"]))
all_features = ohe_features + ["Humidity", "Moisture", "Temperature"]

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": np.round(rf_model.feature_importances_, 4)
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Greens")
plt.title("Feature Importance (Full Model)")
plt.tight_layout()
plt.savefig("results/feature_importance_mapping.png", bbox_inches="tight")
plt.close()

# ==============================================================
# 🔟 Confusion Matrix
# ==============================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix - Full Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix_multi.png", bbox_inches="tight")
plt.close()

# ==============================================================
# 11️⃣ ROC-AUC Curve (Multiclass)
# ==============================================================
print("\n📈 Generating ROC-AUC Curve...")
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_model.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(classes)
roc_auc["macro"] = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
for i, label in enumerate(classes[:10]):  # limit to 10 labels for clarity
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{label} (AUC={roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Multi-Class ROC Curve (Macro AUC={roc_auc["macro"]:.2f})')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/roc_auc_multi.png", bbox_inches="tight")
plt.close()

# ==============================================================
# 12️⃣ Save Mapping and Summary
# ==============================================================
pesticide_map = df.groupby("Plant Type")["pesticide"].first().to_dict()
joblib.dump(pesticide_map, "pesticide_map.joblib")

print("\n💾 Models and artifacts saved successfully!")
print(f"📊 Full model accuracy: {acc*100:.2f}% | CV: {cv*100:.2f}% | MSE: {mse:.4f}")
print(f"🌿 Light model accuracy: {acc_light*100:.2f}% | CV: {cv_light*100:.2f}% | MSE: {mse_light:.4f}")
print("\n📁 All plots saved in /results folder.")
