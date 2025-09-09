import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for wine theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #2c1810 0%, #8b0000 50%, #2c1810 100%);
        color: #f4f4f4;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(139, 0, 0, 0.3);
        border: 2px solid #8b0000;
    }
    
    .stSlider > div > div > div > div {
        background-color: #8b0000;
    }
    
    .wine-header {
        background: linear-gradient(90deg, #8b0000, #dc143c);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
    }
    
    .wine-card {
        background: rgba(139, 0, 0, 0.2);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(220, 20, 60, 0.3);
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .quality-excellent {
        background: linear-gradient(135deg, #228b22, #32cd32);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    .quality-good {
        background: linear-gradient(135deg, #ffa500, #ffd700);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
    }
    
    .quality-average {
        background: linear-gradient(135deg, #ff6347, #ff4500);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    .quality-poor {
        background: linear-gradient(135deg, #8b0000, #dc143c);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_data
def load_model():
    try:
        with open('wine_quality_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running train_model.py")
        return None

# Load sample data for visualization
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('winequality-red.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure winequality-red.csv is in the directory.")
        return None

def main():
    # Header
    st.markdown("""
    <div class="wine-header">
        <h1>🍷 Wine Quality Predictor</h1>
        <p>Predict wine quality based on physicochemical properties</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    sample_data = load_sample_data()
    
    if model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🍇 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "About"])
    
    if page == "Prediction":
        prediction_page(model)
    elif page == "Data Analysis":
        if sample_data is not None:
            analysis_page(sample_data)
    else:
        about_page()

def prediction_page(model):
    st.markdown("### 🎯 Wine Quality Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="wine-card">', unsafe_allow_html=True)
        st.markdown("#### Enter Wine Parameters")
        
        # Input parameters
        col1a, col1b = st.columns(2)
        
        with col1a:
            fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.3, 0.1)
            volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
            residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5, 0.1)
            chlorides = st.slider("Chlorides", 0.01, 0.61, 0.09, 0.001)
            free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15, 1)
        
        with col1b:
            total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46, 1)
            density = st.slider("Density", 0.99, 1.01, 0.997, 0.0001)
            pH = st.slider("pH", 2.7, 4.0, 3.3, 0.01)
            sulphates = st.slider("Sulphates", 0.3, 2.0, 0.7, 0.01)
            alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.4, 0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        if st.button("🔮 Predict Wine Quality", type="primary"):
            # Create feature array
            features = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            
            # Display result in col2
            with col2:
                st.markdown("### 🎯 Prediction Result")
                
                # Quality interpretation
                if prediction >= 7:
                    quality_class = "Excellent"
                    css_class = "quality-excellent"
                elif prediction >= 6:
                    quality_class = "Good"
                    css_class = "quality-good"
                elif prediction >= 5:
                    quality_class = "Average"
                    css_class = "quality-average"
                else:
                    quality_class = "Poor"
                    css_class = "quality-poor"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <h2>Quality Score: {prediction}</h2>
                    <h3>{quality_class} Wine</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### Quality Probability Distribution")
                prob_data = pd.DataFrame({
                    'Quality': range(len(prediction_proba)),
                    'Probability': prediction_proba
                })
                
                fig = px.bar(prob_data, x='Quality', y='Probability', 
                           color='Probability', color_continuous_scale='Reds')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

def analysis_page(data):
    st.markdown("### 📊 Wine Data Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Wines", len(data))
    with col2:
        st.metric("Average Quality", f"{data['quality'].mean():.2f}")
    with col3:
        st.metric("Features", len(data.columns) - 1)
    with col4:
        st.metric("Quality Range", f"{data['quality'].min()}-{data['quality'].max()}")
    
    # Quality distribution
    st.markdown("#### Quality Distribution")
    quality_counts = data['quality'].value_counts().sort_index()
    
    fig1 = px.bar(x=quality_counts.index, y=quality_counts.values, 
                  color=quality_counts.values, color_continuous_scale='Reds',
                  title="Distribution of Wine Quality Scores")
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="Quality Score",
        yaxis_title="Number of Wines"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("#### Feature Correlation Matrix")
    corr_matrix = data.corr()
    
    fig2 = px.imshow(corr_matrix, 
                     color_continuous_scale='RdBu',
                     title="Correlation Between Wine Features")
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Feature analysis
    st.markdown("#### Feature Analysis by Quality")
    
    feature_options = [col for col in data.columns if col != 'quality']
    selected_feature = st.selectbox("Select feature to analyze", feature_options)
    
    # fig3 = px.box(data, x='quality', y=selected_feature, 
    #               color='quality', color_continuous_scale='Reds',
    #               title=f"{selected_feature} by Quality Score")
    # fig3.update_layout(
    #     plot_bgcolor='rgba(0,0,0,0)',
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     font_color='white'
    # )
    # st.plotly_chart(fig3, use_container_width=True)
    fig3 = px.box(data, x='quality', y=selected_feature, 
                color='quality',
                color_discrete_sequence=px.colors.sequential.Reds,
                title=f"{selected_feature} by Quality Score")

    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)


def about_page():
    st.markdown("### 🍷 About Wine Quality Prediction")
    
    st.markdown("""
    <div class="wine-card">
    <h4>🎯 Project Overview</h4>
    <p>This machine learning application predicts wine quality based on physicochemical properties. 
    The model analyzes 11 different features to determine the quality score of red wine.</p>
    
    <h4>📊 Features Used</h4>
    <ul>
        <li><strong>Fixed Acidity:</strong> Acids that don't evaporate readily</li>
        <li><strong>Volatile Acidity:</strong> Amount of acetic acid in wine</li>
        <li><strong>Citric Acid:</strong> Adds freshness and flavor</li>
        <li><strong>Residual Sugar:</strong> Sugar remaining after fermentation</li>
        <li><strong>Chlorides:</strong> Amount of salt in wine</li>
        <li><strong>Free Sulfur Dioxide:</strong> Prevents microbial growth</li>
        <li><strong>Total Sulfur Dioxide:</strong> Total amount of SO2</li>
        <li><strong>Density:</strong> Density of wine relative to water</li>
        <li><strong>pH:</strong> Acidity or basicity of wine</li>
        <li><strong>Sulphates:</strong> Wine additive for antimicrobial properties</li>
        <li><strong>Alcohol:</strong> Percentage of alcohol content</li>
    </ul>
    
    <h4>🎯 Quality Scale</h4>
    <p>Wine quality is rated on a scale of 0-10:</p>
    <ul>
        <li><strong>7-10:</strong> Excellent Quality</li>
        <li><strong>6-7:</strong> Good Quality</li>
        <li><strong>5-6:</strong> Average Quality</li>
        <li><strong>0-5:</strong> Poor Quality</li>
    </ul>
    
    <h4>🛠️ Technology Stack</h4>
    <ul>
        <li>Machine Learning: Random Forest Classifier</li>
        <li>Frontend: Streamlit</li>
        <li>Visualization: Plotly</li>
        <li>Data Processing: Pandas, NumPy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()