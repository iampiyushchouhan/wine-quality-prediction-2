#!/usr/bin/env python3
"""
Wine Quality Prediction - Complete Project Runner
This script handles the complete workflow from setup to running the application.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

class WineQualityProjectRunner:
    """
    Complete project runner for Wine Quality Prediction
    """
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        self.requirements = [
            "streamlit==1.28.1",
            "pandas==2.0.3", 
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "plotly==5.15.0",
            "matplotlib==3.7.2",
            "seaborn==0.12.2"
        ]
        
    def print_banner(self):
        """Print project banner"""
        banner = """
        ┌─────────────────────────────────────────────────────┐
        │            🍷 Wine Quality Predictor 🍷             │
        │                                                     │
        │  Complete ML Project Setup and Runner               │
        │  Built with Python, Streamlit & Scikit-learn       │
        └─────────────────────────────────────────────────────┘
        """
        print(banner)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required!")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    
    def install_requirements(self):
        """Install required packages"""
        print("\n📦 Installing required packages...")
        
        try:
            for package in self.requirements:
                print(f"   Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing packages: {e}")
            print("   Please install manually using: pip install -r requirements.txt")
            return False
    
    def download_dataset(self):
        """Download the wine quality dataset"""
        print("\n🍷 Downloading wine quality dataset...")
        
        dataset_file = self.project_dir / "winequality-red.csv"
        
        if dataset_file.exists():
            print("✅ Dataset already exists!")
            return True
        
        try:
            print("   Downloading from UCI ML Repository...")
            urllib.request.urlretrieve(self.dataset_url, dataset_file)
            
            # Verify the dataset
            df = pd.read_csv(dataset_file, sep=';')  # UCI dataset uses semicolon separator
            print(f"✅ Dataset downloaded! Shape: {df.shape}")
            
            # Convert semicolon-separated to comma-separated for easier processing
            df.to_csv(dataset_file, index=False)
            print("✅ Dataset converted to comma-separated format!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\n📋 Manual download instructions:")
            print("   1. Go to: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009")
            print("   2. Download 'winequality-red.csv'")
            print("   3. Place it in the project directory")
            return False
    
    def verify_files(self):
        """Verify all required files exist"""
        print("\n📁 Verifying project files...")
        
        required_files = [
            "main.py",
            "train_model.py", 
            "data_preprocessing.py",
            "model_evaluation.py",
            "requirements.txt",
            "winequality-red.csv"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = self.project_dir / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - MISSING")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            return False
        
        print("✅ All required files present!")
        return True
    
    def train_model(self):
        """Train the machine learning model"""
        print("\n🤖 Training machine learning model...")
        
        model_file = self.project_dir / "wine_quality_model.pkl"
        
        if model_file.exists():
            user_input = input("   Model already exists. Retrain? (y/N): ").lower()
            if user_input != 'y':
                print("✅ Using existing model!")
                return True
        
        try:
            print("   This may take a few minutes...")
            result = subprocess.run([sys.executable, "train_model.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model trained successfully!")
                print("   Generated files:")
                print("   - wine_quality_model.pkl")
                print("   - scaler.pkl") 
                print("   - confusion_matrix.png")
                print("   - feature_importance.png")
                return True
            else:
                print("❌ Error training model:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running training script: {e}")
            return False
    
    def run_streamlit_app(self):
        """Launch the Streamlit application"""
        print("\n🚀 Launching Streamlit application...")
        print("   The app will open in your default browser.")
        print("   Press Ctrl+C to stop the application.")
        print("\n" + "="*50)
        
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user.")
        except Exception as e:
            print(f"❌ Error launching Streamlit: {e}")
            print("\n📋 Manual launch instructions:")
            print("   1. Open terminal in project directory")
            print("   2. Run: streamlit run main.py")
    
    def setup_project(self):
        """Complete project setup"""
        print("🔧 Setting up Wine Quality Prediction project...")
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Installing requirements", self.install_requirements),
            ("Downloading dataset", self.download_dataset),
            ("Verifying files", self.verify_files),
            ("Training model", self.train_model)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Setup failed at: {step_name}")
                return False
        
        print("\n🎉 Setup completed successfully!")
        return True
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            print("\n" + "="*50)
            print("🍷 Wine Quality Predictor - Main Menu")
            print("="*50)
            print("1. 🔧 Setup Project (First time)")
            print("2. 🤖 Train Model Only")
            print("3. 🚀 Run Application")
            print("4. 📊 Run Data Analysis")
            print("5. 🧪 Test Model Performance")
            print("6. ❌ Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                if self.setup_project():
                    self.run_streamlit_app()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                if (self.project_dir / "wine_quality_model.pkl").exists():
                    self.run_streamlit_app()
                else:
                    print("❌ Model not found! Please setup project first.")
            elif choice == '4':
                self.run_data_analysis()
            elif choice == '5':
                self.run_model_evaluation()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-6.")
    
    def run_data_analysis(self):
        """Run data preprocessing and analysis"""
        print("\n📊 Running data analysis...")
        
        try:
            result = subprocess.run([sys.executable, "data_preprocessing.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Data analysis completed!")
                print("   Check generated visualizations:")
                print("   - data_analysis.png")
            else:
                print("❌ Error in data analysis:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running data analysis: {e}")
    
    def run_model_evaluation(self):
        """Run comprehensive model evaluation"""
        print("\n🧪 Running model evaluation...")
        
        model_file = self.project_dir / "wine_quality_model.pkl"
        
        if not model_file.exists():
            print("❌ Model not found! Please train the model first.")
            return
        
        try:
            result = subprocess.run([sys.executable, "model_evaluation.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model evaluation completed!")
                print("   Check generated reports:")
                print("   - detailed_confusion_matrix.png")
                print("   - detailed_feature_importance.png") 
                print("   - prediction_analysis.png")
                print("   - quality_specific_analysis.png")
            else:
                print("❌ Error in model evaluation:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running model evaluation: {e}")
    
    def quick_start(self):
        """Quick start for experienced users"""
        print("🚀 Quick Start Mode")
        
        if self.setup_project():
            print("\n✨ Setup complete! Launching application...")
            self.run_streamlit_app()
        else:
            print("\n❌ Quick start failed. Please use manual setup.")

def main():
    """Main entry point"""
    runner = WineQualityProjectRunner()
    runner.print_banner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--quick', '-q']:
            runner.quick_start()
        elif arg in ['--setup', '-s']:
            runner.setup_project()
        elif arg in ['--run', '-r']:
            runner.run_streamlit_app()
        elif arg in ['--train', '-t']:
            runner.train_model()
        elif arg in ['--help', '-h']:
            print("Usage: python run.py [option]")
            print("Options:")
            print("  --quick, -q    Quick setup and run")
            print("  --setup, -s    Setup project only")
            print("  --run, -r      Run application only")
            print("  --train, -t    Train model only")
            print("  --help, -h     Show this help")
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for available options")
    else:
        # Interactive menu
        runner.show_menu()

if __name__ == "__main__":
    main()