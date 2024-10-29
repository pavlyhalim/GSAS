# ----------------------------
# Importing Necessary Libraries
# ----------------------------
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder, label_binarize
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier


warnings.filterwarnings('ignore')

# ----------------------------
# Step 1: Data loading
# ----------------------------

def load_data(file_path):
    """
    Loads the data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


# ----------------------------
# Step 2: Data Preprocessing
# ----------------------------

def preprocess_data(df):
    """
    Preprocesses the dataset by handling numerical and categorical features.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame, pd.Series, LabelEncoder: Preprocessed features, target variable, and label encoder.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if 'Career_Path' in numeric_features:
        numeric_features.remove('Career_Path')
    elif 'Career_Path' in categorical_features:
        categorical_features.remove('Career_Path')
    
    y = df['Career_Path']
    X = df.drop('Career_Path', axis=1)

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_numeric = numeric_pipeline.fit_transform(X[numeric_features])

    def preprocess_categorical(X):
        X_cat = X[categorical_features].copy()
        for col in categorical_features:
            X_cat[col] = X_cat[col].astype('category')
        return pd.get_dummies(X_cat, drop_first=True)

    X_categorical = preprocess_categorical(X)

    X_processed = np.hstack([X_numeric, X_categorical.values])

    feature_names = numeric_features + X_categorical.columns.tolist()

    X_preprocessed = pd.DataFrame(X_processed, columns=feature_names)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_preprocessed, y_encoded, le, numeric_pipeline, categorical_features

# ----------------------------
# Step 3: Feature Selection
# ----------------------------

def feature_selection(X, y, k=15):
    """
    Selects top features based on Mutual Information and Recursive Feature Elimination (RFE).

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.

    Returns:
        pd.DataFrame: Selected features.
    """

    def feature_ranking_mutual_info(X, y, k=15):
        """Rank features based on mutual information."""
        mi = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        top_features = mi_scores.head(k).index.tolist()
        return top_features, mi_scores

    top_features_mi, mi_scores = feature_ranking_mutual_info(X, y, k=k)
    print("\nTop Features based on Mutual Information:")
    print(top_features_mi)

    # 3.2: Recursive Feature Elimination (RFE)
    def feature_ranking_rfe(X, y, k=15):
        """Rank features using Recursive Feature Elimination."""
        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=k)
        rfe.fit(X, y)
        rfe_support = rfe.get_support()
        selected_features = X.columns[rfe_support].tolist()
        return selected_features, rfe.ranking_

    top_features_rfe, rfe_ranking = feature_ranking_rfe(X, y, k=k)
    print("\nTop Features based on RFE:")
    print(top_features_rfe)

    # 3.3: Combining Feature Selection Methods
    combined_top_features = list(set(top_features_mi + top_features_rfe))
    print("\nCombined Top Features:")
    print(combined_top_features)

    X_selected = X[combined_top_features]

    return X_selected, combined_top_features

# ----------------------------
# Step 4: Clustering
# ----------------------------

def perform_clustering(X, k=5):
    """
    Applies K-Means clustering to the feature matrix.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        k (int): Number of clusters.

    Returns:
        pd.DataFrame, KMeans: Dataset with cluster assignments, trained KMeans model.
    """
    # Apply K-Means Clustering
    print(f"\nApplying K-Means Clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    X_clustered = X.copy()
    X_clustered['Cluster'] = clusters

    # Visualize clusters using PCA
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X.drop('Cluster', axis=1)) if 'Cluster' in X.columns else pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['Cluster'] = clusters

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=principal_df, palette='viridis', alpha=0.6)
    plt.title('Clusters Visualization using PCA')
    plt.legend(title='Cluster')
    plt.show()

    return X_clustered, kmeans

# ----------------------------
# Step 5: Ranking Career Options
# ----------------------------

def rank_careers(X_clustered, df, feature_weights=None):
    """
    Ranks career options within each cluster based on feature weights.

    Parameters:
        X_clustered (pd.DataFrame): Feature matrix with cluster assignments.
        df (pd.DataFrame): Original dataset.
        feature_weights (dict): Weights assigned to each feature.

    Returns:
        dict: Ranked careers per cluster.
    """
    # Define feature weights based on feature importance or domain knowledge
    if feature_weights is None:
        # assign equal weights for now
        feature_weights = {feature: 1.0 for feature in X_clustered.columns if feature != 'Cluster'}

    def score_career(row, weights):
        """
        Assigns a score to each career option based on weighted features.

        Parameters:
            row (pd.Series): A row of feature values.
            weights (dict): A dictionary of feature weights.

        Returns:
            float: The computed score.
        """
        score = 0.0
        for feature, weight in weights.items():
            score += row.get(feature, 0) * weight
        return score

    # Apply scoring within each cluster
    ranked_careers = {}
    clusters = X_clustered['Cluster'].unique()
    for cluster in clusters:
        cluster_data = X_clustered[X_clustered['Cluster'] == cluster]
        if cluster_data.empty:
            continue
        cluster_scores = cluster_data.apply(lambda row: score_career(row, feature_weights), axis=1)
        ranked_indices = cluster_scores.sort_values(ascending=False).index
        ranked_careers[cluster] = ranked_indices.tolist()

    for cluster, indices in ranked_careers.items():
        print(f"\nCluster {cluster} Ranked Careers:")
        for idx in indices:
            print(f"Student ID: {df.loc[idx, 'Student_ID']} - Career Path: {df.loc[idx, 'Career_Path']}")

    return ranked_careers

# ----------------------------
# Step 6: Classification
# ----------------------------

def classification_models(X_selected, y_encoded, label_encoder, numeric_pipeline, X_categorical_columns, cv_folds=5):
    """
    Trains and evaluates classification models (Random Forest and SVM).

    Parameters:
        X_selected (pd.DataFrame): Selected features.
        y_encoded (pd.Series): Encoded target variable.
        label_encoder (LabelEncoder): Label encoder instance.
        numeric_pipeline (Pipeline): Pipeline used for numerical features.
        X_categorical_columns (list): List of categorical column names.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        RandomForestClassifier, SVC: Trained classifiers.
    """
    # Split the data with stratification
    print("\nSplitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded,
        test_size=0.2, stratify=y_encoded, random_state=42
    )

    print("\nClass Distribution in Training Set:")
    print(pd.Series(y_train).value_counts())

    print("\nClass Distribution in Testing Set:")
    print(pd.Series(y_test).value_counts())

    rf_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42, probability=True)

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Define hyperparameter grids
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }

    # Hyperparameter Tuning for Random Forest
    print("\nTuning Random Forest Classifier...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best Random Forest Parameters: {rf_grid.best_params_}")

    # Hyperparameter Tuning for SVM
    print("\nTuning SVM Classifier...")
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        svm_params,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1
    )

    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    print(f"Best SVM Parameters: {svm_grid.best_params_}")

    print("\nTraining classifiers with best parameters...")
    best_rf.fit(X_train, y_train)
    best_svm.fit(X_train, y_train)

    y_pred_rf = best_rf.predict(X_test)
    y_pred_svm = best_svm.predict(X_test)

    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)
    y_pred_svm_decoded = label_encoder.inverse_transform(y_pred_svm)

    # Evaluation Metrics
    def evaluate_model(y_true, y_pred, model_name='Model'):
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_true, y_pred, zero_division=0))
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"{model_name} Accuracy: {acc * 100:.2f}%")
        print(f"{model_name} Precision: {prec * 100:.2f}%")
        print(f"{model_name} Recall: {rec * 100:.2f}%")
        print(f"{model_name} F1 Score: {f1 * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

    evaluate_model(y_test_decoded, y_pred_rf_decoded, model_name='Random Forest')

    evaluate_model(y_test_decoded, y_pred_svm_decoded, model_name='SVM')

    return best_rf, best_svm

# ----------------------------
# Step 7: Evaluation
# ----------------------------

def evaluate_models(best_rf, best_svm, X_test, y_test, label_encoder, model_name=None):
    """
    Computes additional evaluation metrics like Mean Average Precision (MAP).

    Parameters:
        best_rf (RandomForestClassifier): Trained Random Forest classifier.
        best_svm (SVC): Trained SVM classifier.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True labels.
        label_encoder (LabelEncoder): Label encoder instance.
        model_name (str, optional): Name of the model being evaluated.
    """
    y_test_binarized = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
    n_classes = y_test_binarized.shape[1]

    y_prob_rf = best_rf.predict_proba(X_test)

    if n_classes > 2:
        map_score = average_precision_score(y_test_binarized, y_prob_rf, average="macro")
    else:
        map_score = average_precision_score(y_test_binarized, y_prob_rf, average="macro")

    model_name = model_name or "Random Forest"
    print(f"\n{model_name} MAP Score: {map_score:.2f}")
# ----------------------------
# Step 8: Simulation and Validation
# ----------------------------

def simulation_validation(X_test, y_test, y_pred_rf, y_pred_svm, df):
    """
    Simulates user feedback and validates recommendations.

    Parameters:
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True labels.
        y_pred_rf (np.array): Random Forest predictions.
        y_pred_svm (np.array): SVM predictions.
        df (pd.DataFrame): Original dataset.
    """
    simulation_results = pd.DataFrame({
        'Student_ID': df.loc[X_test.index, 'Student_ID'],
        'Actual_Career_Path': y_test,
        'Recommended_Career_Path_RF': y_pred_rf,
        'Recommended_Career_Path_SVM': y_pred_svm
    })

    print("\nSimulation Results:")
    print(simulation_results.head())

# ----------------------------
# Step 9: Visualization of Feature Importance
# ----------------------------

def visualize_feature_importance(best_rf, X_selected):
    """
    Visualizes the top 15 feature importances from the Random Forest model.

    Parameters:
        best_rf (RandomForestClassifier): Trained Random Forest classifier.
        X_selected (pd.DataFrame): Selected feature matrix.
    """
    importances = best_rf.feature_importances_
    feature_importances = pd.Series(importances, index=X_selected.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances[:15], y=feature_importances.index[:15], palette='viridis')
    plt.title('Top 15 Feature Importances from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

# ----------------------------
# Step 10: Saving the Model
# ----------------------------

def save_models(best_rf, best_svm, numeric_pipeline, label_encoder):
    """
    Saves the trained models and preprocessors using joblib.
    """
    print("\nSaving models and preprocessors...")
    joblib.dump(best_rf, 'best_random_forest_model.joblib')
    joblib.dump(best_rf.get_params(), 'rf_best_params.joblib')
    joblib.dump(best_svm, 'best_svm_model.joblib')
    joblib.dump(best_svm.get_params(), 'svm_best_params.joblib')
    joblib.dump(numeric_pipeline, 'numeric_pipeline.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    print("Models and preprocessors have been saved successfully.")

# ----------------------------
# Step 11: Cross Validate Techniques
# ----------------------------

def cross_validate_model(best_rf, X_selected, y_encoded, cv_folds=5):
    """
    Performs cross-validation on the Random Forest model.

    Parameters:
        best_rf (RandomForestClassifier): Trained Random Forest classifier.
        X_selected (pd.DataFrame): Selected feature matrix.
        y_encoded (pd.Series): Encoded target variable.
        cv_folds (int): Number of cross-validation folds.
    """
    print("\nPerforming Cross-Validation for Random Forest...")
    cv_scores = cross_val_score(best_rf, X_selected, y_encoded, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")

# ----------------------------
# Step 12: Handling Class Imbalance
# ----------------------------

def handle_class_imbalance(X_train, y_train, class_weight='balanced'):
    """
    Handles class imbalance by computing class weights.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.
        class_weight (str or dict): How to weight classes.

    Returns:
        dict: Computed class weights.
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight=class_weight, classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    print("\nComputed Class Weights:")
    print(class_weights_dict)
    return class_weights_dict

def train_weighted_random_forest(X_train, y_train, class_weights):
    """
    Trains a weighted Random Forest classifier to handle class imbalance.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.
        class_weights (dict): Class weights.

    Returns:
        RandomForestClassifier: Trained Random Forest classifier.
    """
    print("\nTraining Weighted Random Forest Classifier...")
    rf_weighted = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        class_weight=class_weights,
        random_state=42
    )
    rf_weighted.fit(X_train, y_train)
    return rf_weighted


##############################
###TEST
##############################


# def improved_feature_engineering(X):
#     # Add interaction terms
#     X['GPA_TestScore_Interaction'] = X['GPA'] * X['Test_Scores']
    
#     # Add domain-specific features (example)
#     X['Tech_Orientation'] = X['Skill_Programming'] + X['Skill_Data Analysis'] + X['Interest_Technology']
    
#     return X

# def improved_model_selection(X_train, y_train):
#     # XGBoost
#     xgb_model = xgb.XGBClassifier(random_state=42)
#     xgb_params = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.3],
#         'n_estimators': [100, 200, 300],
#         'min_child_weight': [1, 3, 5]
#     }
#     xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=20, cv=5, random_state=42, n_jobs=-1)
#     xgb_search.fit(X_train, y_train)
    
#     return xgb_search.best_estimator_

# def handle_imbalance(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# def improved_evaluation(y_true, y_pred):
#     kappa = cohen_kappa_score(y_true, y_pred)
#     mcc = matthews_corrcoef(y_true, y_pred)
#     print(f"Cohen's Kappa: {kappa:.2f}")
#     print(f"Matthews Correlation Coefficient: {mcc:.2f}")


# ----------------------------
# Main Execution
# ----------------------------

# if __name__ == "__main__":
#     # Step 1: Generate Synthetic Data
#     print("Generating synthetic dataset...")
#     df_synthetic = generate_synthetic_data(num_samples=10000, random_state=42)

#     # Optionally, save the synthetic dataset
#     df_synthetic.to_csv('synthetic_career_planning_data.csv', index=False)

#     # Step 2: Preprocess Data
#     print("\nPreprocessing data...")
#     X_preprocessed, y_encoded, label_encoder, numeric_pipeline, X_categorical_columns = preprocess_data(df_synthetic)

#     # Step 3: Improved Feature Engineering
#     print("\nPerforming improved feature engineering...")
#     X_engineered = improved_feature_engineering(X_preprocessed)

#     # Step 4: Feature Selection
#     print("\nSelecting top features...")
#     X_selected, combined_top_features = feature_selection(X_engineered, y_encoded, k=20)  # Increased k to 20

#     # Step 5: Dimensionality Reduction
#     print("\nApplying dimensionality reduction...")
#     pca = PCA(n_components=0.95)  # Preserve 95% of variance
#     X_pca = pca.fit_transform(X_selected)
#     print(f"Reduced dimensions from {X_selected.shape[1]} to {X_pca.shape[1]}")

#     # Step 6: Handle Class Imbalance
#     print("\nHandling class imbalance...")
#     X_resampled, y_resampled = handle_imbalance(X_pca, y_encoded)

#     # Step 7: Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
#     )

#     # Step 8: Improved Model Selection and Training
#     print("\nTraining improved models...")
#     xgb_model = improved_model_selection(X_train, y_train)
#     rf_model = BalancedRandomForestClassifier(n_estimators=200, random_state=42)
#     rf_model.fit(X_train, y_train)

#     # Step 9: Model Evaluation
#     print("\nEvaluating models...")
#     for name, model in [("XGBoost", xgb_model), ("Balanced Random Forest", rf_model)]:
#         y_pred = model.predict(X_test)
#         y_pred_decoded = label_encoder.inverse_transform(y_pred)
#         y_test_decoded = label_encoder.inverse_transform(y_test)
        
#         print(f"\nEvaluation for {name}:")
#         print(classification_report(y_test_decoded, y_pred_decoded))
#         improved_evaluation(y_test_decoded, y_pred_decoded)

#     # Step 10: Feature Importance Visualization
#     # print("\nVisualizing feature importance...")
#     # visualize_feature_importance(rf_model, X_selected.columns)

#     # Step 11: Clustering (optional, if still relevant)
#     print("\nPerforming clustering...")
#     X_clustered, kmeans_model = perform_clustering(X_pca, k=5)

#     # Step 12: Career Ranking within Clusters
#     print("\nRanking careers within clusters...")
#     ranked_careers = rank_careers(X_clustered, df_synthetic, feature_weights=None)

#     # Step 13: Cross-Validation
#     print("\nPerforming cross-validation...")
#     cross_validate_model(rf_model, X_resampled, y_resampled, cv_folds=5)

#     # Step 14: Simulation and Validation
#     print("\nRunning simulation and validation...")
#     simulation_validation(X_test, y_test, xgb_model.predict(X_test), rf_model.predict(X_test), df_synthetic)

#     # Step 15: Save Models
#     print("\nSaving models...")
#     save_models(rf_model, xgb_model, numeric_pipeline, label_encoder)

#     print("\nImproved SRFcML Model Implementation Complete.")

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    print("Loading dataset...")
    df_real = load_data('path_to_your_real_data.csv')  #file path

    print("\nPreprocessing data...")
    X_preprocessed, y_encoded, label_encoder, numeric_pipeline, categorical_features = preprocess_data(df_real)

    print("\nSelecting top features...")
    X_selected, combined_top_features = feature_selection(X_preprocessed, y_encoded, k=15)

    optimal_k = 5  # Optimal number of clusters
    X_clustered, kmeans_model = perform_clustering(X_selected, k=optimal_k)

    print("\nRanking career options within clusters...")
    ranked_careers = rank_careers(X_clustered, df_real, feature_weights=None)

    print("\nTraining and evaluating classification models...")
    best_rf, best_svm = classification_models(X_selected, y_encoded, label_encoder, numeric_pipeline, categorical_features, cv_folds=5)

    print("\nEvaluating additional metrics...")
    evaluate_models(best_rf, best_svm, X_selected, y_encoded, label_encoder)

    simulation_validation(X_selected, y_encoded, best_rf.predict(X_selected), best_svm.predict(X_selected), df_real)

    visualize_feature_importance(best_rf, X_selected)

    save_models(best_rf, best_svm, numeric_pipeline, label_encoder)

    cross_validate_model(best_rf, X_selected, y_encoded, cv_folds=5)

    class_weights = handle_class_imbalance(X_selected, y_encoded, class_weight='balanced')
    rf_weighted = train_weighted_random_forest(X_selected, y_encoded, class_weights)

    y_pred_weighted = rf_weighted.predict(X_selected)
    y_pred_weighted_decoded = label_encoder.inverse_transform(y_pred_weighted)

    print("\nEvaluation of Weighted Random Forest Classifier:")
    evaluate_models(rf_weighted, best_svm, X_selected, y_encoded, label_encoder, model_name='Random Forest Weighted')

    print("\nSRFcML Model Implementation Complete.")