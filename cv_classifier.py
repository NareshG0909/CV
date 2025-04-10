import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
csv_path = '/Users/apple/CV/TrashType_Image_Dataset/results/features.csv'
df = pd.read_csv(csv_path)

# --- Check class distribution ---
print("\n📌 Class Distribution:\n")
print(df['label'].value_counts())

# --- Features and labels ---
X = df[['color_score', 'edge_strength', 'mean_intensity', 'corner_strength']]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# --- Train-test split with scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Models to test ---
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga')
}

# --- Collect results ---
results = []
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })

    # Save confusion matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# --- Summary Table ---
summary_df = pd.DataFrame(results)
print("\n📊 Model Performance Summary:\n")
print(summary_df.to_string(index=False))

# --- Plot confusion matrices ---
for name, cm in conf_matrices.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
f