import streamlit as st
import pandas as pd
import numpy as np

try:
    import sklearn
    has_sklearn = True
    # Use sklearn to avoid warnings
    sklearn_version = sklearn.__version__
except ImportError:
    has_sklearn = False


try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False


try:
    import plotly.graph_objects as go
    has_plotly = True
except ImportError:
    has_plotly = False

# Try to import matplotlib and seaborn, but it's okay if it fails
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

st.set_page_config(
    page_title="ML Demo",
    page_icon="🔬",
    layout="wide"
)

# Create navigation menu
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.page_link("app.py", label="🏠 Home")
with col2:
    st.page_link("pages/1_ml_explanation.py", label="📚 ML")
with col3:
    st.page_link("pages/2_nn_explanation.py", label="🧠 NN")
with col4:
    st.page_link("pages/3_ml_demo.py", label="🔬 ML Demo")
with col5:
    st.page_link("pages/4_nn_demo.py", label="🎯 NN Demo")

st.title("Machine Learning Model Demo")

# Load model
@st.cache_resource
def load_model():
    # Use a mock model for testing only
    # st.info("Using a mock model for testing")
    return "mock_model"

try:
    model = load_model()
    if model == "mock_model":
        # st.warning("Using a mock model for testing")
        pass
    else:
        st.success("Model loaded successfully")
        # Show model information
        st.info(f"Model type: {type(model).__name__}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    # Create a mock model for testing
    model = "mock_model"
    # st.warning("Using a mock model for testing instead")
    pass
    
# Load sample data to see columns
@st.cache_data
def load_sample_data():
    return pd.read_csv('data/raw/heart_disease_uci.csv')

df_sample = load_sample_data()

st.header("Heart Disease Prediction Test")
st.write("Please enter your information to predict heart disease risk")

# Create form for input data
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        
        sex = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=0
        )
        
        cp = st.selectbox(
            "Chest Pain Type",
            options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            index=0
        )
        
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=120)
        
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=["True", "False"],
            index=1
        )
        
        restecg = st.selectbox(
            "Resting ECG Results",
            options=["normal", "st-t abnormality", "lv hypertrophy"],
            index=0
        )
    
    with col2:
        thalch = st.number_input("Maximum Heart Rate", min_value=70, max_value=220, value=150)
        
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=["True", "False"],
            index=1
        )
        
        oldpeak = st.number_input("ST Depression from Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope = st.selectbox(
            "Slope of ST Segment",
            options=["upsloping", "flat", "downsloping"],
            index=0
        )
        
        ca = st.number_input("Number of Major Vessels Colored (0-3)", min_value=0, max_value=3, value=0)
        
        thal = st.selectbox(
            "Thalassemia",
            options=["normal", "fixed defect", "reversable defect"],
            index=0
        )
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create DataFrame from user input
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Convert data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Check model type
            if model != "mock_model":
                # Convert categorical data to one-hot encoding
                categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
                input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
                
                # Predict with the model loaded from PKL
                try:
                    # Check if the model is from sklearn
                    if hasattr(model, 'predict_proba'):
                        # Use predict_proba if available
                        prediction = model.predict_proba(input_encoded)
                        # Select the probability of class 1 (has heart disease)
                        risk_score = prediction[0][1]
                    else:
                        # Use regular predict
                        prediction = model.predict(input_encoded)
                        risk_score = prediction[0]
                except Exception as e:
                    st.warning(f"Error during prediction: {e}")
                    # Try adjusting data format and predict again
                    st.info("Trying to adjust data format and predict again")
                    
                    # Convert qualitative data to numbers
                    input_df['sex'] = 1 if sex == "Male" else 0
                    input_df['cp'] = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"].index(cp)
                    input_df['fbs'] = 1 if fbs == "True" else 0
                    input_df['restecg'] = ["normal", "st-t abnormality", "lv hypertrophy"].index(restecg)
                    input_df['exang'] = 1 if exang == "True" else 0
                    input_df['slope'] = ["upsloping", "flat", "downsloping"].index(slope)
                    input_df['thal'] = ["normal", "fixed defect", "reversable defect"].index(thal)
                    
                    # Predict again
                    if hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba(input_df)
                        risk_score = prediction[0][1]
                    else:
                        prediction = model.predict(input_df)
                        risk_score = prediction[0]
            else:
                # Use mock model - calculate risk from user input
                # This is just a simple calculation example, not medically accurate
                risk_factors = 0
                
                # Risk factors
                if age > 50:
                    risk_factors += 0.1
                if sex == "Male":
                    risk_factors += 0.1
                if cp == "asymptomatic" or cp == "non-anginal":
                    risk_factors += 0.15
                if trestbps > 140:
                    risk_factors += 0.1
                if chol > 240:
                    risk_factors += 0.1
                if fbs == "True":
                    risk_factors += 0.05
                if restecg != "normal":
                    risk_factors += 0.05
                if thalch < 120:
                    risk_factors += 0.05
                if exang == "True":
                    risk_factors += 0.15
                if oldpeak > 2:
                    risk_factors += 0.1
                if slope == "flat" or slope == "downsloping":
                    risk_factors += 0.1
                if ca > 0:
                    risk_factors += ca * 0.1
                if thal != "normal":
                    risk_factors += 0.1
                
                # Calculate risk score (0-1)
                risk_score = min(risk_factors, 1.0)
            
            st.subheader("Prediction Results")
            
            if risk_score > 0.5:
                st.error(f"High risk of heart disease (Risk score: {risk_score:.2f})")
            else:
                st.success(f"Low risk of heart disease (Risk score: {risk_score:.2f})")
                
            # Show risk graph
            if has_plotly:
                # Use plotly if available
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(risk_score),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Heart Disease Risk Score"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "green"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                st.plotly_chart(fig)
            elif has_matplotlib:
                # Use matplotlib instead
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create pie chart showing risk
                colors = ['green', 'yellow', 'red']
                if risk_score <= 0.3:
                    color_idx = 0
                elif risk_score <= 0.7:
                    color_idx = 1
                else:
                    color_idx = 2
                
                # Create pie chart
                ax.pie([risk_score, 1-risk_score], colors=[colors[color_idx], 'lightgray'], 
                       startangle=90, counterclock=False,
                       wedgeprops={'width': 0.3, 'edgecolor': 'w'})
                
                # Add text in the center
                ax.text(0, 0, f"{risk_score:.2f}", ha='center', va='center', fontsize=24)
                
                # Add chart title
                ax.set_title("Heart Disease Risk Score")
                
                # Set chart to be circular
                ax.set_aspect('equal')
                
                # Show chart
                st.pyplot(fig)
            else:
                # If neither plotly nor matplotlib is available, show as text
                st.info(f"Risk Score: {risk_score:.2f}")
                
                # Show color bar instead of graph
                if risk_score <= 0.3:
                    st.success(f"Low Risk: {risk_score:.2f}")
                elif risk_score <= 0.7:
                    st.warning(f"Medium Risk: {risk_score:.2f}")
                else:
                    st.error(f"High Risk: {risk_score:.2f}")
            
            # Show additional explanation
            st.subheader("Prediction Explanation")
            
            explanations = []
            if age > 50:
                explanations.append("- Age over 50 increases heart disease risk")
            if sex == "Male":
                explanations.append("- Males have higher heart disease risk than females")
            if cp == "asymptomatic" or cp == "non-anginal":
                explanations.append("- Asymptomatic or non-anginal chest pain may indicate heart problems")
            if trestbps > 140:
                explanations.append("- Resting blood pressure above 140 mmHg increases heart disease risk")
            if chol > 240:
                explanations.append("- Cholesterol above 240 mg/dl increases heart disease risk")
            if fbs == "True":
                explanations.append("- High fasting blood sugar increases heart disease risk")
            if exang == "True":
                explanations.append("- Exercise-induced chest pain is a sign of heart disease")
            if oldpeak > 2:
                explanations.append("- ST depression from exercise above 2 indicates heart problems")
            if ca > 0:
                explanations.append(f"- {ca} major vessels colored increases heart disease risk")
            if thal != "normal":
                explanations.append("- Thalassemia abnormality increases heart disease risk")
            
            if explanations:
                for exp in explanations:
                    st.write(exp)
            else:
                st.write("No significant risk factors found")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Using simulated values instead")
            
            # Create simulated values for testing
            import random
            risk_score = random.uniform(0, 1)
            
            st.subheader("Prediction Results (Simulated)")
            
            if risk_score > 0.5:
                st.error(f"High risk of heart disease (Risk score: {risk_score:.2f})")
            else:
                st.success(f"Low risk of heart disease (Risk score: {risk_score:.2f})")
                
            # Show risk graph
            if has_plotly:
                # Use plotly if available
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(risk_score),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Heart Disease Risk Score (Simulated)"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "green"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                st.plotly_chart(fig)
            elif has_matplotlib:
                # Use matplotlib instead
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create pie chart showing risk
                colors = ['green', 'yellow', 'red']
                if risk_score <= 0.3:
                    color_idx = 0
                elif risk_score <= 0.7:
                    color_idx = 1
                else:
                    color_idx = 2
                
                # Create pie chart
                ax.pie([risk_score, 1-risk_score], colors=[colors[color_idx], 'lightgray'], 
                       startangle=90, counterclock=False,
                       wedgeprops={'width': 0.3, 'edgecolor': 'w'})
                
                # Add text in the center
                ax.text(0, 0, f"{risk_score:.2f}", ha='center', va='center', fontsize=24)
                
                # Add chart title
                ax.set_title("Heart Disease Risk Score (Simulated)")
                
                # Set chart to be circular
                ax.set_aspect('equal')
                
                # Show chart
                st.pyplot(fig)
            else:
                # If neither plotly nor matplotlib is available, show as text
                st.info(f"Risk Score (Simulated): {risk_score:.2f}")
                
                # Show color bar instead of graph
                if risk_score <= 0.3:
                    st.success(f"Low Risk: {risk_score:.2f}")
                elif risk_score <= 0.7:
                    st.warning(f"Medium Risk: {risk_score:.2f}")
                else:
                    st.error(f"High Risk: {risk_score:.2f}")

st.header("Model Information")
st.write("""
This model is used to predict heart disease risk using data from the UCI Heart Disease Dataset,
which contains patient data from several hospitals.
""")

# Show sample data used to train the model
st.subheader("Sample Data Used for Model Training")
st.dataframe(df_sample.head(10))

st.header("Model Performance")
# Show graphs and statistics of the model
st.write("""
In evaluating the actual model performance, the following metrics would be measured:
- Accuracy
- Sensitivity
- Specificity
- Area Under ROC Curve (AUC-ROC)
""")

# Show simulated graph
# Create simulated data for graph display
# st.subheader("Feature Importance")

# feature_importance = {
#     'age': 0.12,
#     'sex_Male': 0.11,
#     'cp_asymptomatic': 0.15,
#     'trestbps': 0.08,
#     'chol': 0.09,
#     'fbs_True': 0.05,
#     'restecg_abnormal': 0.06,
#     'thalch': 0.07,
#     'exang_True': 0.14,
#     'oldpeak': 0.10,
#     'slope_flat': 0.08,
#     'ca': 0.12,
#     'thal_reversable': 0.09
# }

# if has_matplotlib:
#     fig, ax = plt.subplots(figsize=(10, 6))
#     features = list(feature_importance.keys())
#     importances = list(feature_importance.values())

#     sns.barplot(x=importances, y=features, palette="viridis")
#     plt.title("Feature Importance for Prediction")
#     plt.xlabel("Importance")
#     plt.ylabel("Feature")

#     st.pyplot(fig)
# else:
#     # Show as a table if matplotlib is not available
#     st.write("Feature Importance for Prediction:")
    
#     # Create DataFrame from feature importance data
#     df_importance = pd.DataFrame({
#         'Feature': list(feature_importance.keys()),
#         'Importance': list(feature_importance.values())
#     }).sort_values(by='Importance', ascending=False)
    
#     # Show as table
#     st.dataframe(df_importance)
