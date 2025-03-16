import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os

# อ่านข้อมูล
df = pd.read_csv('data/raw/Nvidia_stock.csv')

# แปลงคอลัมน์ Date เป็น datetime
df['Date'] = pd.to_datetime(df['Date'])

# แปลงคอลัมน์ราคาจากสตริงเป็นตัวเลข (ลบเครื่องหมาย $ และแปลงเป็น float)
price_columns = ['Close/Last', 'Open', 'High', 'Low']
for col in price_columns:
    df[col] = df[col].str.replace('$', '').astype(float)

# สร้างฟีเจอร์เพิ่มเติม
df['MA5'] = df['Close/Last'].rolling(window=5).mean()  # Moving average 5 วัน
df['MA20'] = df['Close/Last'].rolling(window=20).mean()  # Moving average 20 วัน
df['Price_Change'] = df['Close/Last'].pct_change()  # การเปลี่ยนแปลงราคา

# ลบแถวที่มีค่า NaN
df = df.dropna()

# เตรียมข้อมูลสำหรับโมเดล
features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'Price_Change']
X = df[features]
y = df['Close/Last']

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สเกลข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างและฝึกสอนโมเดล
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = {}

for name, model in models.items():
    # ฝึกสอนโมเดล
    model.fit(X_train_scaled, y_train)
    
    # ทำนายผล
    y_pred = model.predict(X_test_scaled)
    
    # คำนวณ metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'R2': r2
    }

# แสดงผลลัพธ์
print("\nผลการทำนายของแต่ละโมเดล:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")

# สร้างกราฟเปรียบเทียบ RMSE
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [metrics['RMSE'] for metrics in results.values()])
plt.title('เปรียบเทียบค่า RMSE ของแต่ละโมเดล')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# เลือกโมเดลที่ดีที่สุด (ตามค่า RMSE)
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
print(f"\nBest Model: {best_model_name}")
print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"R2 Score: {results[best_model_name]['R2']:.4f}")


