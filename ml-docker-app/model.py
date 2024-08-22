from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib
from xgboost import XGBClassifier

# Load the data
digits = datasets.load_digits()
X = pd.DataFrame(digits.data)
y = pd.DataFrame(digits.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for XGBoost
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Set up the randomized search with cross-validation
random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, n_jobs=-1, random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
print(f"Best parameters: {random_search.best_params_}")

# Use the best estimator to predict
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Evaluate the model
print(f"Improved Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the tuned model
best_xgb.get_booster().save_model('xgb_tuned.json')