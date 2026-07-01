import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create output folder for figures if it doesn't exist
os.makedirs("static/images", exist_ok=True)

print("="*50)
print("STARTING MACHINE LEARNING PIPELINE")
print("="*50)

# 1. LOAD DATA
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv("Sales - Marketing customer dataset.csv")
print(f"Dataset shape: {df.shape}")

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n[STEP 2] Running EDA...")

# Info and descriptive stats
print("\nDescriptive Statistics:")
desc_stats = df.describe(include='all').transpose()
desc_stats.to_csv("static/eda_summary.csv")
print(df.head())

# 2.1. Missing values bar chart
plt.figure(figsize=(10, 5))
missing_pct = df.isnull().sum() / len(df) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_pct) > 0:
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='viridis')
    plt.title("Persentase Missing Values per Kolom (%)")
    plt.xlabel("Persentase (%)")
    plt.ylabel("Kolom")
    plt.tight_layout()
    plt.savefig("static/images/missing_values.png")
    plt.close()
    print("Saved missing_values.png")
else:
    print("No missing values found.")

# 2.2. Target Churn distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df, palette='Set2')
plt.title("Distribusi Target (Churn)")
plt.xlabel("Churn (0 = Tidak, 1 = Ya)")
plt.ylabel("Jumlah Pelanggan")
plt.tight_layout()
plt.savefig("static/images/churn_distribution.png")
plt.close()
print("Saved churn_distribution.png")

# 2.3. Correlation Heatmap for numerical features
plt.figure(figsize=(12, 10))
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, annot_kws={"size": 8})
plt.title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")


# 3. DATA SPLITTING & PREPROCESSING HELPER
# Helper to perform feature engineering (tenure)
def feature_engineering(data):
    data = data.copy()
    
    # Parse dates
    signup = pd.to_datetime(data['signup_date'], errors='coerce')
    last_purchase = pd.to_datetime(data['last_purchase_date'], errors='coerce')
    
    # Calculate tenure in days
    data['tenure_days'] = (last_purchase - signup).dt.days
    # Fill negative or null tenure days with 0
    data['tenure_days'] = data['tenure_days'].clip(lower=0)
    
    # Drop irrelevant features
    cols_to_drop = ['customer_id', 'signup_date', 'last_purchase_date']
    data = data.drop(columns=cols_to_drop, errors='ignore')
    return data

# Preprocess the entire dataframe for dates first
df_feat = feature_engineering(df)

# Define X and y
X = df_feat.drop(columns=['churn'])
y = df_feat['churn']

# Identify numerical and categorical columns
numeric_features = ['age', 'total_visits', 'avg_session_time', 'pages_per_session', 
                    'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value', 
                    'support_tickets', 'delivery_delay_days', 'satisfaction_score', 'nps_score', 
                    'marketing_spend_per_user', 'lifetime_value', 'last_3_month_purchase_freq', 
                    'tenure_days', 'is_premium_user', 'discount_used', 'refund_requested']

categorical_features = ['gender', 'country', 'city', 'acquisition_channel', 
                        'device_type', 'subscription_type', 'coupon_code', 'payment_method']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


# Helper to calculate and format classification metrics
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n[{name} Evaluation]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Confusion_Matrix": cm.tolist()
    }


# 4. EXPERIMENT 1: DIRECT MODELING (Minimal conversion, no scaling/outlier/tuning)
print("\n" + "="*50)
print("[STEP 3] EXPERIMENT 1: DIRECT MODELING")
print("="*50)

# We use simple pipeline with imputation and one-hot encoding, but NO SCALING
direct_preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# Fit preprocessor on X_train
X_train_dir = direct_preprocessor.fit_transform(X_train)
X_test_dir = direct_preprocessor.transform(X_test)

# Models for Direct scenario
lr_dir = LogisticRegression(max_iter=1000, random_state=42)
rf_dir = RandomForestClassifier(random_state=42)

# Voting classifier components
knn_voting = KNeighborsClassifier()
svm_voting = SVC(probability=True, random_state=42, max_iter=1000) # limit max_iter to prevent slow run
voting_dir = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, random_state=42, max_iter=1000)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'
)

# Train and evaluate
results_dir = []

lr_dir.fit(X_train_dir, y_train)
results_dir.append(evaluate_model("Logistic Regression (Direct)", y_test, lr_dir.predict(X_test_dir)))

rf_dir.fit(X_train_dir, y_train)
results_dir.append(evaluate_model("Random Forest (Direct)", y_test, rf_dir.predict(X_test_dir)))

voting_dir.fit(X_train_dir, y_train)
results_dir.append(evaluate_model("Voting Classifier (Direct)", y_test, voting_dir.predict(X_test_dir)))


# 5. EXPERIMENT 2: PREPROCESSED MODELING (Scaling and outlier handling)
print("\n" + "="*50)
print("[STEP 4] EXPERIMENT 2: PREPROCESSED MODELING")
print("="*50)

# Outlier handling: Let's create a custom function to cap outliers in training data
# We only cap specific skewed numeric features
def cap_outliers(df, cols, lower_pct=1, upper_pct=99):
    df_capped = df.copy()
    for col in cols:
        if col in df_capped.columns:
            lower = np.percentile(df_capped[col].dropna(), lower_pct)
            upper = np.percentile(df_capped[col].dropna(), upper_pct)
            df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    return df_capped

# Cap numerical outliers on train and test
X_train_prep = cap_outliers(X_train, numeric_features)
X_test_prep = cap_outliers(X_test, numeric_features)

# Pipeline for preprocessed scenario: Imputation + Encoding + Scaling
prep_transformer = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

X_train_prep_trans = prep_transformer.fit_transform(X_train_prep)
X_test_prep_trans = prep_transformer.transform(X_test_prep)

# Models for Preprocessed scenario
lr_prep = LogisticRegression(max_iter=1000, random_state=42)
rf_prep = RandomForestClassifier(random_state=42)
voting_prep = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, random_state=42, max_iter=2000)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'
)

# Train and evaluate
results_prep = []

lr_prep.fit(X_train_prep_trans, y_train)
results_prep.append(evaluate_model("Logistic Regression (Prep)", y_test, lr_prep.predict(X_test_prep_trans)))

rf_prep.fit(X_train_prep_trans, y_train)
results_prep.append(evaluate_model("Random Forest (Prep)", y_test, rf_prep.predict(X_test_prep_trans)))

voting_prep.fit(X_train_prep_trans, y_train)
results_prep.append(evaluate_model("Voting Classifier (Prep)", y_test, voting_prep.predict(X_test_prep_trans)))


# 6. EXPERIMENT 3: HYPERPARAMETER TUNING & FEATURE SELECTION
print("\n" + "="*50)
print("[STEP 5] EXPERIMENT 3: HYPERPARAMETER TUNING & FEATURE SELECTION")
print("="*50)

# Feature Importance from Random Forest (Preprocessed)
# We can get names after column transformer
encoded_cat_names = prep_transformer.named_transformers_['cat'].named_steps['encode'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(encoded_cat_names)

importances = rf_prep.feature_importances_
feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
feat_importances.head(20).plot(kind='barh', color='teal')
plt.title("Top 20 Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("static/images/feature_importances.png")
plt.close()
print("Saved feature_importances.png")

# Grid Search on Random Forest (Preprocessed)
print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train_prep_trans, y_train)
best_rf = grid_search_rf.best_estimator_
print(f"Best RF Parameters: {grid_search_rf.best_params_}")

# Grid Search on Logistic Regression
print("\nTuning Logistic Regression...")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2']
}
grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=lr_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search_lr.fit(X_train_prep_trans, y_train)
best_lr = grid_search_lr.best_estimator_
print(f"Best LR Parameters: {grid_search_lr.best_params_}")

# Re-evaluate Tuned models
results_tuned = []
results_tuned.append(evaluate_model("Tuned Logistic Regression", y_test, best_lr.predict(X_test_prep_trans)))
results_tuned.append(evaluate_model("Tuned Random Forest", y_test, best_rf.predict(X_test_prep_trans)))

# Compare and pick the absolute best model
all_results = results_dir + results_prep + results_tuned
results_df = pd.DataFrame(all_results)
results_df.to_csv("static/model_comparison.csv", index=False)
print("\n--- ALL EXPERIMENT RESULTS ---")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])

# Find model with best F1-score
best_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
best_model_name = best_row['Model']
print(f"\nBest Model selected: {best_model_name} with F1-Score of {best_row['F1-Score']:.4f}")

# 7. SAVE THE BEST PIPELINE
# To make it deployable, we save the full pipeline. Let's build a clean Pipeline with the best estimator
if "Random Forest" in best_model_name:
    final_classifier = best_rf
elif "Logistic Regression" in best_model_name:
    final_classifier = best_lr
else:
    # default to tuned random forest
    final_classifier = best_rf

# Construct the full final pipeline (which does preprocessing AND classification)
final_pipeline = Pipeline(steps=[
    ('preprocessor', prep_transformer),
    ('classifier', final_classifier)
])

# Save the pipeline and helper variables
model_data = {
    'pipeline': final_pipeline,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names,
    'cap_limits': {
        col: (np.percentile(X_train[col].dropna(), 1), np.percentile(X_train[col].dropna(), 99))
        for col in numeric_features if col in X_train.columns
    }
}

joblib.dump(model_data, "best_churn_pipeline.joblib")
print("\n[SUCCESS] Saved the best pipeline to 'best_churn_pipeline.joblib'")
print("Done!")
