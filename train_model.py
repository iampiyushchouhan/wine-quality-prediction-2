import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the wine dataset"""
    print("Loading wine quality dataset...")
    
    # Load the dataset
    data = pd.read_csv('winequality-red.csv')
    
    print(f"Dataset shape: {data.shape}")
    print("\nDataset info:")
    print(data.info())
    print("\nFirst few rows:")
    print(data.head())
    print("\nStatistical summary:")
    print(data.describe())
    print("\nQuality distribution:")
    print(data['quality'].value_counts().sort_index())
    
    # Check for missing values
    print(f"\nMissing values: {data.isnull().sum().sum()}")
    
    return data

def preprocess_data(data):
    """Preprocess the data for training"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_model(X_train, y_train):
    """Train the Random Forest model with hyperparameter tuning"""
    print("\nTraining Random Forest model...")
    
    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier()
    
    # Perform grid search
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=sorted(set(y_test)), 
                yticklabels=sorted(set(y_test)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Quality')
    plt.ylabel('Actual Quality')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='Reds_r')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_accuracy

def save_model(model, scaler, feature_names):
    """Save the trained model, scaler, and feature names"""
    print("\nSaving model...")
    
    # Create a dictionary with all necessary components
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    # Save to pickle file
    with open('wine_quality_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    print("Model saved as 'wine_quality_model.pkl'")
    
    # Save scaler separately
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    print("Scaler saved as 'scaler.pkl'")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Wine Quality Prediction - Model Training")
    print("=" * 60)
    
    try:
        # Load and explore data
        data = load_and_explore_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        
        # Save model
        save_model(model, scaler, feature_names)
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {accuracy:.4f}")
        print("Files created:")
        print("- wine_quality_model.pkl")
        print("- scaler.pkl")
        print("- confusion_matrix.png")
        print("- feature_importance.png")
        
    except FileNotFoundError:
        print("Error: 'winequality-red.csv' not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009")
        print("Or from UCI ML Repository")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()