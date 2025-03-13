import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix Matplotlib GUI issue
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Store dataset and model in memory
df_memory = None
model_memory = None
feature_columns = []
categorical_columns = {}
numeric_ranges = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df_memory, feature_columns, categorical_columns, numeric_ranges
    file = request.files['file']
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    df_memory = pd.read_csv(file)
    
    if 'Attrition' not in df_memory.columns:
        return jsonify({"error": "Dataset must contain an 'Attrition' column"}), 400
    
    feature_columns = [col for col in df_memory.columns if col != 'Attrition']
    
    # Identify categorical columns (≤ 10 unique values) for dropdowns
    categorical_columns = {
        col: df_memory[col].dropna().unique().tolist()
        for col in df_memory.select_dtypes(include=['object']).columns
        if df_memory[col].nunique() <= 10
    }
    
    # Identify numerical columns and get value ranges
    numeric_ranges = {
        col: {"min": df_memory[col].min(), "max": df_memory[col].max()}
        for col in df_memory.select_dtypes(include=[np.number]).columns
    }
    
    sample_data = df_memory.head(10).to_dict(orient='records')
    
    return jsonify({
        "message": "File uploaded successfully!",
        "sample_data": sample_data,
        "features": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_ranges": numeric_ranges
    })

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     global df_memory
#     if df_memory is None:
#         return jsonify({"error": "No file uploaded"}), 400

#     df = df_memory.copy()

#     for col in df.columns:
#         if df[col].isnull().sum() > 0:
#             if df[col].dtype == np.number:
#                 df[col] = df[col].fillna(df[col].mean())
#             else:
#                 df[col] = df[col].fillna(df[col].mode()[0])

#     df = df.drop_duplicates()

#     label_encoders = {}
#     for col in df.select_dtypes(include=['object']).columns:
#         if df[col].nunique() <= 10:
#             label_encoders[col] = LabelEncoder()
#             df[col] = label_encoders[col].fit_transform(df[col])

#     df_memory = df
#     return jsonify({"message": "EDA completed successfully!"})
@app.route('/analyze', methods=['POST'])
def analyze():
    global df_memory
    if df_memory is None:
        return jsonify({"error": "No file uploaded"}), 400

    df = df_memory.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == np.number:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() <= 10:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

    df_memory = df
    return jsonify({"message": "EDA completed successfully!"})


# @app.route('/train', methods=['POST'])
# def train_model():
#     global df_memory, model_memory
#     if df_memory is None:
#         return jsonify({"error": "No file uploaded"}), 400

#     model_choice = request.form.get('model_choice')

#     if model_choice not in ['logistic', 'random_forest', 'decision_tree']:
#         return jsonify({"error": "Invalid model selection"}), 400

#     df = df_memory.copy()
#     X = df.drop(columns=['Attrition'], errors='ignore')
#     y = df['Attrition']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     if model_choice == 'logistic':
#         model = LogisticRegression()
#     elif model_choice == 'random_forest':
#         model = RandomForestClassifier()
#     elif model_choice == 'decision_tree':
#         model = DecisionTreeClassifier()

#     model.fit(X_train, y_train)
#     model_memory = model
    
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     return jsonify({"message": f"Model trained successfully! Accuracy: {accuracy:.2f}"})
from imblearn.over_sampling import SMOTE

@app.route('/train', methods=['POST'])
def train_model():
    global df_memory, model_memory
    if df_memory is None:
        return jsonify({"error": "No file uploaded"}), 400

    model_choice = request.form.get('model_choice')

    if model_choice not in ['logistic', 'random_forest', 'decision_tree']:
        return jsonify({"error": "Invalid model selection"}), 400

    df = df_memory.copy()
    
    # Convert categorical target variable to numerical
    if df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Attrition'], errors='ignore')
    y = df['Attrition']

    # Splitting dataset into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # **Apply SMOTE for Oversampling**
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Select model
    if model_choice == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_choice == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=10, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    model_memory = model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return jsonify({"message": f"Model trained successfully! Accuracy: {accuracy:.4f}"})


@app.route('/predict', methods=['POST'])
def predict():
    global model_memory, feature_columns
    if model_memory is None:
        return jsonify({"error": "No trained model found!"}), 400

    input_data = {}
    for col in feature_columns:
        value = request.form.get(col)
        if value is not None:
            if col in categorical_columns:
                input_data[col] = value  # Keep categorical values as selected
            else:
                input_data[col] = float(value)  # Convert numeric values to float
    
    df_input = pd.DataFrame([input_data])

    # Apply EDA to the new data sample
    for col in df_input.columns:
        if df_input[col].isnull().sum() > 0:
            if df_input[col].dtype == np.number:
                df_input[col] = df_input[col].fillna(df_input[col].mean())
            else:
                df_input[col] = df_input[col].fillna(df_input[col].mode()[0])

    label_encoders = {}
    for col in df_input.select_dtypes(include=['object']).columns:
        if col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            df_input[col] = label_encoders[col].fit_transform(df_input[col])
    
    prediction = model_memory.predict(df_input)[0]
    prediction_label = "Yes" if prediction == 1 else "No"
    
    return jsonify({"prediction": prediction_label, "details": input_data})

if __name__ == '__main__':
    app.run(debug=True)
