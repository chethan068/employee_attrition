import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def train_attrition_model(df, model_choice='random_forest'):
    """
    Train a machine learning model for employee attrition prediction.
    
    Args:
        df (DataFrame): Input DataFrame with employee data
        model_choice (str): Model type to train ('logistic', 'random_forest', or 'decision_tree')
        
    Returns:
        tuple: (model, scaler, metrics, feature_importances)
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure target variable is properly formatted
    if 'Attrition' in df.columns:
        if df['Attrition'].dtype == object:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("Attrition column not found in dataset")
    
    # Separate features and target
    X = df.drop(columns=['Attrition'], errors='ignore')
    y = df['Attrition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train selected model
    if model_choice == 'logistic':
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif model_choice == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Get feature importances
    feature_importances = {}
    if hasattr(model, 'coef_'):
        # For linear models like logistic regression
        importance = np.abs(model.coef_[0])
        for i, col in enumerate(X.columns):
            feature_importances[col] = float(importance[i])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        for i, col in enumerate(X.columns):
            feature_importances[col] = float(model.feature_importances_[i])
    
    # Save model
    model_path = f'models/attrition_model.pkl'
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = f'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    print(f"Model trained: {model_choice}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return model, scaler, metrics, feature_importances



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import joblib
# from imblearn.over_sampling import SMOTE
# from sklearn.pipeline import Pipeline


# def train_attrition_model(df, model_choice='random_forest'):
#     """
#     Train a machine learning model for employee attrition prediction with improved performance.
#     """
#     df = df.copy()
    
#     if 'Attrition' in df.columns:
#         if df['Attrition'].dtype == object:
#             df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
#     else:
#         raise ValueError("Attrition column not found in dataset")
    
#     X = df.drop(columns=['Attrition'], errors='ignore')
#     y = df['Attrition']
    
#     # Handle categorical variables
#     categorical_cols = X.select_dtypes(include=['object']).columns
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
#     X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
#     X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
#     X.drop(columns=categorical_cols, inplace=True)
#     X = pd.concat([X, X_encoded], axis=1)
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     # Handle class imbalance with SMOTE
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Model selection and hyperparameter tuning
#     if model_choice == 'random_forest':
#         model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
#     elif model_choice == 'gradient_boosting':
#         model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
#     else:
#         raise ValueError(f"Invalid model choice: {model_choice}")
    
#     model.fit(X_train_scaled, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test_scaled)
    
#     # Metrics
#     metrics = {
#         'accuracy': float(accuracy_score(y_test, y_pred)),
#         'precision': float(precision_score(y_test, y_pred)),
#         'recall': float(recall_score(y_test, y_pred)),
#         'f1_score': float(f1_score(y_test, y_pred)),
#         'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
#     }
    
#     # Save model and scaler
#     joblib.dump(model, 'models/attrition_model.pkl')
#     joblib.dump(scaler, 'models/scaler.pkl')
    
#     return model, scaler, metrics
