import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(
    page_title="Neural Network Demo",
    page_icon="🎯",
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


# ฟังก์ชันแบ่งข้อมูลแทน train_test_split
def my_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    test_size_count = int(n * test_size)
    indices = np.random.permutation(n)
    test_idx, train_idx = indices[:test_size_count], indices[test_size_count:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test

# ฟังก์ชัน StandardScaler แบบง่าย
class MyStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std
    
    def transform(self, X):
        return (X - self.mean) / self.std

# โมเดล Linear Regression แบบง่าย
class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self
    
    def predict(self, X):
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        return X_with_intercept @ np.append(self.intercept_, self.coef_)

# โมเดลง่ายๆ แทน RandomForest และ SVR
class MySimpleModel:
    def __init__(self, name="SimpleModel"):
        self.name = name
        self.w = None
    
    def fit(self, X, y):
        # เพียงแค่หาค่าเฉลี่ยถ่วงน้ำหนักง่ายๆ
        self.w = np.random.random(X.shape[1])
        self.w = self.w / np.sum(self.w)
        self.bias = np.mean(y)
        return self
    
    def predict(self, X):
        return np.dot(X, self.w) + self.bias

# ฟังก์ชันวัดประสิทธิภาพ
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

# Function to train and save models
def train_and_save_models():
    # Read data
    df = pd.read_csv('data/raw/Nvidia_stock.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert price columns to float
    price_columns = ['Close/Last', 'Open', 'High', 'Low']
    for col in price_columns:
        df[col] = df[col].str.replace('$', '').astype(float)
    
    # Create additional features
    df['MA5'] = df['Close/Last'].rolling(window=5).mean()
    df['MA20'] = df['Close/Last'].rolling(window=20).mean()
    df['Price_Change'] = df['Close/Last'].pct_change()
    
    # Drop NaN values
    df = df.dropna()
    
    # Prepare data
    features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'Price_Change']
    X = df[features]
    y = df['Close/Last']
    
    # Split data
    X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = MyStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': MyLinearRegression(),
        'Random Forest': MySimpleModel("Random Forest"),
        'SVR': MySimpleModel("SVR")
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        
        # Save each model
        os.makedirs('models', exist_ok=True)
        with open(f'models/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return results

# Load models and scaler
@st.cache_resource
def load_models_and_scaler():
    models = {}
    for name in ['linear_regression', 'random_forest', 'svr']:
        try:
            with open(f'models/{name}_model.pkl', 'rb') as f:
                models[name.replace('_', ' ').title()] = pickle.load(f)
        except:
            # ถ้าไม่พบไฟล์โมเดล สร้าง dummy models
            if name == 'linear_regression':
                models[name.replace('_', ' ').title()] = MyLinearRegression()
            else:
                models[name.replace('_', ' ').title()] = MySimpleModel(name.replace('_', ' ').title())
    
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = MyStandardScaler()
    
    return models, scaler

# Main Streamlit app
def main():
    st.title("Neural Network Model Demo")
    
    # Check if models exist, if not train them
    if not os.path.exists('models/random_forest_model.pkl'):
        st.write("Training models... This may take a moment.")
        results = train_and_save_models()
        st.write("Models trained and saved successfully!")
    else:
        results = None
    
    # Load models and scaler
    models, scaler = load_models_and_scaler()
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox(
        "Choose a model",
        options=list(models.keys()),
        index=0  # Default to Linear Regression
    )
    
    # Input form
    st.header("Enter Stock Data")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            open_price = st.number_input("Open Price ($)", min_value=0.0, value=100.0)
            high_price = st.number_input("High Price ($)", min_value=0.0, value=102.0)
            low_price = st.number_input("Low Price ($)", min_value=0.0, value=98.0)
            volume = st.number_input("Volume", min_value=0, value=1000000)
        
        with col2:
            ma5 = st.number_input("5-day Moving Average ($)", min_value=0.0, value=100.0)
            ma20 = st.number_input("20-day Moving Average ($)", min_value=0.0, value=95.0)
            price_change = st.number_input("Price Change (%)", value=0.0, format="%.4f")
        
        submit = st.form_submit_button("Predict Closing Price")
    
    # Prediction
    if submit:
        # Prepare input data
        input_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Volume': [volume],
            'MA5': [ma5],
            'MA20': [ma20],
            'Price_Change': [price_change]
        })
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        selected_model = models[model_name]
        prediction = selected_model.predict(input_scaled)[0]
        
        # Display result
        st.success(f"Predicted Closing Price: ${prediction:.2f}")
        
        # Show model performance if available
        if results:
            st.write(f"{model_name} Performance:")
            st.write(f"RMSE: {results[model_name]['RMSE']:.2f}")
            st.write(f"R2 Score: {results[model_name]['R2']:.4f}")
    
    # ลบส่วนการพล็อตกราฟออก

if __name__ == "__main__":
    # Train models if not exists and run app
    main()
    