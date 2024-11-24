import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.io import arff
import joblib

data, meta = arff.loadarff('chessAnalysis.arff')
df = pd.DataFrame(data)

for column in df.select_dtypes([object]).columns:
    df[column] = df[column].str.decode('utf-8')

print(f"Data shape: {df.shape}")
print("\nInitial data preview:")
print(df.head())

X = df.drop('class', axis=1)
y = df['class'].replace({'won': 1, 'nowin': 0})

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Min samples at a leaf node
    'max_features': ['sqrt', 'log2', None]  # Max features to consider
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=skf, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters found:", best_params)
best_rf = grid_search.best_estimator_
y_test_pred = best_rf.predict(X_test)
y_train_pred = best_rf.predict(X_train)


print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))


cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cannot Win', 'Can Win'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc:.2f}")


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


feature_importances = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X_encoded.columns[sorted_idx])
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.show()

joblib.dump(best_rf, 'best_rf_model.pkl')
