# Databricks notebook source

# COMMAND ----------

# Basic Libraries
import pandas as pd
import numpy as np
import math
import warnings

# Data Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier

# Oversampling for Imbalanced Data
from imblearn.over_sampling import SMOTE

# Kaggle Integration
import kagglehub

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning)


# COMMAND ----------


class DepressionAnalysis:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.model = None

    def preprocess_data(self):
        print("Checking for NaN values in the dataset...")
        print(self.df.isnull().sum())

        # Fill missing values with median
        if self.df.isnull().sum().any():
            print("Handling NaN values by filling with median...")
            self.df.fillna(self.df.median(), inplace=True)

        # Encoding income categories and SES feature
        self.df['Income Category'] = pd.cut(self.df['Income'], bins=[0, 30000, 70000, self.df['Income'].max()],
                                            labels=['Low', 'Middle', 'High'])
        self.df['SES'] = self.df['Income Category'].astype(str) + "_" + self.df['Employment Status']
        
        # Encode categorical columns
        categorical_columns = ['Marital Status', 'Education Level', 'Smoking Status', 'Physical Activity Level',
                               'Employment Status', 'Alcohol Consumption', 'Dietary Habits', 'Sleep Patterns',
                               'History of Substance Abuse', 'Family History of Depression', 
                               'Chronic Medical Conditions', 'Income Category', 'SES']
        
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = LabelEncoder().fit_transform(self.df[col])

        print("Data Preprocessed")

    def feature_engineering(self):
        # Health Risk Score and Wellness Index
        self.df['Health Risk Score'] = (
            self.df['Smoking Status'].map({'Non-smoker': 0, 'Former': 1, 'Smoker': 2}) +
            self.df['Alcohol Consumption'].map({'Low': 0, 'Moderate': 1, 'High': 2}) +
            self.df['Chronic Medical Conditions'].apply(lambda x: 1 if x == 'Yes' else 0)
        )
        self.df['Wellness Index'] = (
            self.df['Physical Activity Level'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2}) * 0.4 +
            self.df['Dietary Habits'].map({'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}) * 0.3 +
            self.df['Sleep Patterns'].map({'Poor': 0, 'Fair': 1, 'Good': 2}) * 0.3
        )
        print("Feature Engineering Completed")

    def eda(self):
        print("Dataset Info:")
        print(self.df.info())
        
        print("\nSummary Statistics:")
        print(self.df.describe())
        
        # Distribution of Numerical Features
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)
        num_rows = math.ceil(num_cols / 3) 
        fig = make_subplots(rows=num_rows, cols=3, subplot_titles=numerical_cols)
        
        for i, col in enumerate(numerical_cols):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1
            fig.add_trace(go.Histogram(x=self.df[col], name=col), row=row, col=col_pos)
            
        fig.update_layout(title="Distribution of Numerical Features", height=600, showlegend=False)
        fig.show()
        
        # Distribution of Categorical Variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            fig = px.histogram(self.df, x=col)
            fig.update_layout(title=f'Distribution of {col}')
            fig.show()
        
        # Target Variable Analysis
        print("\nTarget Variable Distribution:")
        target_counts = self.df['History of Mental Illness'].value_counts()
        fig = px.bar(x=target_counts.index, y=target_counts.values, labels={'x': 'History of Mental Illness', 'y': 'Count'})
        fig.update_layout(title="Class Distribution in Target Variable")
        fig.show()
        
        # Correlation Matrix
        correlation_matrix = self.df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(title="Correlation Matrix")
        fig.show()
        
        # Outlier Detection with Boxplots
        for col in numerical_cols:
            fig = px.box(self.df, y=col, title=f'Boxplot of {col}')
            fig.show()
        
        # Pairplot Analysis for Bivariate Relationships
        important_features = ['Income', 'Physical Activity Level', 'Dietary Habits', 'Sleep Patterns','History of Mental Illness']
        fig = px.scatter_matrix(self.df[important_features], dimensions=important_features, color='History of Mental Illness')
        fig.update_layout(title='Pairplot of Important Features')
        fig.show()

        print("EDA Completed")

    def handle_imbalance(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Data Balanced using SMOTE")
        return X_resampled, y_resampled

    def split_data(self):
        X = self.df.drop(['Name', 'History of Mental Illness'], axis=1)
        y = LabelEncoder().fit_transform(self.df['History of Mental Illness'])
        
        print("Checking for NaN values in X and y before splitting...")
        print("X NaN values:", X.isnull().sum().any())
        
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def tune_model(self, X_train, y_train):
        xgb = XGBClassifier(eval_metric='logloss')

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print("Model Tuned")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("AUC Score:", roc_auc_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    def feature_importance(self, X_train):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices])
        plt.title("Feature Importance")
        plt.show()

    def cross_val_evaluation(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        print("Cross-validated AUC Scores:", scores)
        print("Average AUC Score:", scores.mean())

    def full_analysis(self):
        self.preprocess_data()
        # self.feature_engineering()
        self.eda()
        
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # Tune model with balanced data if desired
        self.tune_model(X_train_balanced, y_train_balanced)
        
        # Evaluate model with original test set
        self.evaluate_model(X_test, y_test)
        self.feature_importance(X_train)
        self.cross_val_evaluation(X_train, y_train)


# COMMAND ----------

# Download latest version
path = kagglehub.dataset_download("anthonytherrien/depression-dataset")

print("Path to dataset files:", path)

# Assuming the dataset is a CSV file
dataset_path = os.path.join(path, 'depression_data.csv')  # Replace 'filename.csv' with the actual file name

# Load the dataset into a DataFrame
# depression_data_df = pd.read_csv(dataset_path)


# COMMAND ----------

analysis = DepressionAnalysis(dataset_path)
analysis.full_analysis()

# COMMAND ----------

dataset_path = os.path.join(path, 'depression_data.csv')

analysis = DepressionAnalysis(dataset_path)
analysis.full_analysis()
