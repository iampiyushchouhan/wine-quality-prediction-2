import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class WineDataProcessor:
    """
    A comprehensive data processor for wine quality dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """Load wine dataset from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"File {filepath} not found!")
            return None
    
    def explore_data(self, data):
        """Comprehensive data exploration"""
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        # Data types
        print("\nData types:")
        print(data.dtypes)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(data.describe())
        
        # Quality distribution
        print("\nQuality Distribution:")
        quality_dist = data['quality'].value_counts().sort_index()
        print(quality_dist)
        
        # Correlation with quality
        print("\nCorrelation with Quality:")
        correlations = data.corr()['quality'].sort_values(ascending=False)
        print(correlations)
        
        return {
            'shape': data.shape,
            'missing_values': data.isnull().sum().sum(),
            'quality_distribution': quality_dist,
            'correlations': correlations
        }
    
    def visualize_data(self, data):
        """Create visualizations for data understanding"""
        # Set style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Quality distribution
        plt.subplot(3, 4, 1)
        data['quality'].value_counts().sort_index().plot(kind='bar', color='darkred')
        plt.title('Quality Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        
        # Correlation heatmap
        plt.subplot(3, 4, 2)
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix')
        
        # Distribution of key features
        key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
        
        for i, feature in enumerate(key_features, 3):
            plt.subplot(3, 4, i)
            data[feature].hist(bins=30, color='darkred', alpha=0.7)
            plt.title(f'Distribution of {feature.title()}')
            plt.xlabel(feature.title())
            plt.ylabel('Frequency')
        
        # Box plots for features vs quality
        key_features_box = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'pH']
        
        for i, feature in enumerate(key_features_box, 7):
            plt.subplot(3, 4, i)
            data.boxplot(column=feature, by='quality', ax=plt.gca())
            plt.title(f'{feature.title()} by Quality')
            plt.suptitle('')  # Remove auto title
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def clean_data(self, data):
        """Clean and preprocess the data"""
        print("\nCleaning data...")
        
        # Make a copy
        cleaned_data = data.copy()
        
        # Remove duplicates
        initial_shape = cleaned_data.shape
        cleaned_data = cleaned_data.drop_duplicates()
        print(f"Removed {initial_shape[0] - cleaned_data.shape[0]} duplicate rows")
        
        # Handle outliers using IQR method
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'quality']
        
        for column in numeric_columns:
            Q1 = cleaned_data[column].quantile(0.25)
            Q3 = cleaned_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(cleaned_data[(cleaned_data[column] < lower_bound) | 
                                             (cleaned_data[column] > upper_bound)])
            
            # Cap outliers instead of removing them
            cleaned_data[column] = np.clip(cleaned_data[column], lower_bound, upper_bound)
            
            if outliers_before > 0:
                print(f"Capped {outliers_before} outliers in {column}")
        
        return cleaned_data
    
    def feature_engineering(self, data):
        """Create new features from existing ones"""
        print("\nEngineering features...")
        
        engineered_data = data.copy()
        
        # Create new features
        engineered_data['acidity_ratio'] = (engineered_data['fixed acidity'] / 
                                           engineered_data['volatile acidity'])
        
        engineered_data['sulfur_dioxide_ratio'] = (engineered_data['free sulfur dioxide'] / 
                                                  engineered_data['total sulfur dioxide'])
        
        engineered_data['alcohol_acidity_interaction'] = (engineered_data['alcohol'] * 
                                                         engineered_data['fixed acidity'])
        
        # Categorize quality into groups
        engineered_data['quality_category'] = pd.cut(
            engineered_data['quality'], 
            bins=[0, 4, 6, 8, 10], 
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        print(f"Added {len(engineered_data.columns) - len(data.columns)} new features")
        
        return engineered_data
    
    def prepare_for_ml(self, data):
        """Prepare data for machine learning"""
        print("\nPreparing data for ML...")
        
        # Separate features and target
        feature_columns = [col for col in data.columns if col not in ['quality', 'quality_category']]
        X = data[feature_columns]
        y = data['quality']
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Feature names: {list(X.columns)}")
        
        return X, y, feature_columns
    
    def save_processed_data(self, data, filename='processed_wine_data.csv'):
        """Save processed data to CSV"""
        data.to_csv(filename, index=False)
        print(f"Processed data saved to {filename}")

def main():
    """Main data processing pipeline"""
    print("Wine Quality Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = WineDataProcessor()
    
    # Load data
    data = processor.load_data('winequality-red.csv')
    
    if data is not None:
        # Explore data
        stats = processor.explore_data(data)
        
        # Visualize data
        processor.visualize_data(data)
        
        # Clean data
        cleaned_data = processor.clean_data(data)
        
        # Feature engineering
        engineered_data = processor.feature_engineering(cleaned_data)
        
        # Prepare for ML
        X, y, feature_names = processor.prepare_for_ml(engineered_data)
        
        # Save processed data
        processor.save_processed_data(engineered_data)
        
        print("\nData processing completed successfully!")
        print(f"Final dataset shape: {engineered_data.shape}")
        
        return engineered_data, X, y, feature_names
    
    else:
        print("Data processing failed!")
        return None, None, None, None

if __name__ == "__main__":
    main()