import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.model_selection import learning_curve, validation_curve
import pickle
import warnings
warnings.filterwarnings('ignore')

class WineModelEvaluator:
    """
    Comprehensive model evaluation for wine quality prediction
    """
    
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_model_and_data(self, model_path='wine_quality_model.pkl'):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded successfully from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_path} not found!")
            return False
    
    def set_test_data(self, X_test, y_test):
        """Set test data for evaluation"""
        self.X_test = X_test
        self.y_test = y_test
        
        # Make predictions
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)
        
        print(f"Test data set. Shape: {X_test.shape}")
    
    def basic_metrics(self):
        """Calculate and display basic metrics"""
        print("=" * 50)
        print("BASIC METRICS")
        print("=" * 50)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        # Precision, Recall, F1-score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_pred, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Quality': sorted(set(self.y_test)),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print("\nDetailed Metrics by Quality:")
        print(metrics_df)
        
        return accuracy, metrics_df
    
    def confusion_matrix_analysis(self):
        """Detailed confusion matrix analysis"""
        print("\n" + "=" * 50)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 50)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                   xticklabels=sorted(set(self.y_test)),
                   yticklabels=sorted(set(self.y_test)))
        axes[0].set_title('Confusion Matrix (Raw Counts)')
        axes[0].set_xlabel('Predicted Quality')
        axes[0].set_ylabel('Actual Quality')
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
                   xticklabels=sorted(set(self.y_test)),
                   yticklabels=sorted(set(self.y_test)))
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted Quality')
        axes[1].set_ylabel('Actual Quality')
        
        plt.tight_layout()
        plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analysis
        print(f"Total predictions: {cm.sum()}")
        print(f"Correct predictions: {np.trace(cm)}")
        print(f"Incorrect predictions: {cm.sum() - np.trace(cm)}")
        
        return cm, cm_normalized
    
    def feature_importance_analysis(self, feature_names):
        """Analyze feature importance"""
        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("Feature Importance Ranking:")
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', 
                       palette='Reds_r')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('detailed_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        else:
            print("Model doesn't support feature importance analysis")
            return None
    
    def prediction_distribution_analysis(self):
        """Analyze prediction distributions"""
        print("\n" + "=" * 50)
        print("PREDICTION DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Actual vs Predicted distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual distribution
        axes[0, 0].hist(self.y_test, bins=range(3, 9), alpha=0.7, color='blue', 
                       edgecolor='black', label='Actual')
        axes[0, 0].set_title('Actual Quality Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Predicted distribution
        axes[0, 1].hist(self.y_pred, bins=range(3, 9), alpha=0.7, color='red', 
                       edgecolor='black', label='Predicted')
        axes[0, 1].set_title('Predicted Quality Distribution')
        axes[0, 1].set_xlabel('Quality Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Comparison
        axes[1, 0].hist(self.y_test, bins=range(3, 9), alpha=0.5, color='blue', 
                       label='Actual', edgecolor='black')
        axes[1, 0].hist(self.y_pred, bins=range(3, 9), alpha=0.5, color='red', 
                       label='Predicted', edgecolor='black')
        axes[1, 0].set_title('Actual vs Predicted Distribution')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Prediction errors
        errors = self.y_pred - self.y_test
        axes[1, 1].hist(errors, bins=range(-4, 5), alpha=0.7, color='orange', 
                       edgecolor='black')
        axes[1, 1].set_title('Prediction Errors Distribution')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', label='Perfect Prediction')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Error analysis
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Square Error: {rmse:.4f}")
        print(f"Error Standard Deviation: {np.std(errors):.4f}")
        
        return errors, mae, rmse
    
    def quality_specific_analysis(self):
        """Analyze performance for each quality level"""
        print("\n" + "=" * 50)
        print("QUALITY-SPECIFIC ANALYSIS")
        print("=" * 50)
        
        quality_analysis = []
        
        for quality in sorted(set(self.y_test)):
            # Get indices for this quality
            quality_indices = self.y_test == quality
            
            if np.sum(quality_indices) > 0:
                # Actual count
                actual_count = np.sum(quality_indices)
                
                # Predicted count
                predicted_count = np.sum(self.y_pred == quality)
                
                # Correctly predicted count
                correct_count = np.sum((self.y_test == quality) & (self.y_pred == quality))
                
                # Calculate metrics
                precision = correct_count / predicted_count if predicted_count > 0 else 0
                recall = correct_count / actual_count if actual_count > 0 else 0
                
                quality_analysis.append({
                    'Quality': quality,
                    'Actual_Count': actual_count,
                    'Predicted_Count': predicted_count,
                    'Correct_Count': correct_count,
                    'Precision': precision,
                    'Recall': recall
                })
        
        analysis_df = pd.DataFrame(quality_analysis)
        print(analysis_df)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Counts comparison
        x = analysis_df['Quality']
        axes[0].bar(x - 0.2, analysis_df['Actual_Count'], 0.4, label='Actual', alpha=0.7)
        axes[0].bar(x + 0.2, analysis_df['Predicted_Count'], 0.4, label='Predicted', alpha=0.7)
        axes[0].set_title('Actual vs Predicted Counts by Quality')
        axes[0].set_xlabel('Quality Score')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        
        # Precision by quality
        axes[1].bar(analysis_df['Quality'], analysis_df['Precision'], color='green', alpha=0.7)
        axes[1].set_title('Precision by Quality Score')
        axes[1].set_xlabel('Quality Score')
        axes[1].set_ylabel('Precision')
        
        # Recall by quality
        axes[2].bar(analysis_df['Quality'], analysis_df['Recall'], color='blue', alpha=0.7)
        axes[2].set_title('Recall by Quality Score')
        axes[2].set_xlabel('Quality Score')
        axes[2].set_ylabel('Recall')
        
        plt.tight_layout()
        plt.savefig('quality_specific_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return analysis_df
    
    def generate_comprehensive_report(self, feature_names):
        """Generate a comprehensive evaluation report"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Basic metrics
        accuracy, metrics_df = self.basic_metrics()
        
        # Confusion matrix
        cm, cm_norm = self.confusion_matrix_analysis()
        
        # Feature importance
        importance_df = self.feature_importance_analysis(feature_names)
        
        # Prediction analysis
        errors, mae, rmse = self.prediction_distribution_analysis()
        
        # Quality-specific analysis
        quality_df = self.quality_specific_analysis()
        
        # Summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Square Error: {rmse:.4f}")
        print(f"Total Test Samples: {len(self.y_test)}")
        
        if importance_df is not None:
            print(f"\nTop 3 Important Features:")
            for i, row in importance_df.head(3).iterrows():
                print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\nBest Performing Quality Scores:")
        best_quality = quality_df.loc[quality_df['Precision'].idxmax(), 'Quality']
        print(f"  Highest Precision: Quality {best_quality}")
        
        best_recall_quality = quality_df.loc[quality_df['Recall'].idxmax(), 'Quality']
        print(f"  Highest Recall: Quality {best_recall_quality}")
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'metrics_df': metrics_df,
            'importance_df': importance_df,
            'quality_df': quality_df
        }

def main():
    """Main evaluation pipeline"""
    evaluator = WineModelEvaluator()
    
    # Load model
    if evaluator.load_model_and_data():
        print("Model evaluation requires test data.")
        print("Please run this after training the model with train_model.py")
    
if __name__ == "__main__":
    main()