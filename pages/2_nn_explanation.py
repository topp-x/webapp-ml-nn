import streamlit as st

st.set_page_config(
    page_title="NN Explanation",
    page_icon="🧠",
    layout="wide"
)

# สร้างเมนูนำทาง
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.page_link("app.py", label="🏠 หน้าหลัก")
with col2:
    st.page_link("pages/1_ml_explanation.py", label="📚 ML")
with col3:
    st.page_link("pages/2_nn_explanation.py", label="🧠 NN")
with col4:
    st.page_link("pages/3_ml_demo.py", label="🔬 ML Demo")
with col5:
    st.page_link("pages/4_nn_demo.py", label="🎯 NN Demo")
st.header("การอธิบายโมเดล Neural Network")
st.write("ไฟล์นี้เป็นชุดข้อมูลเกี่ยวกับราคาหุ้น NVIDIA โดยดูจากปัจจัยต่าง ๆ ที่อาจมีผลต่อราคาหุ้น NVIDIA จากแหล่งข้อมูล NVIDIA Common Stock Dataset ซึ่งเป็นหนึ่งในชุดข้อมูลยอดนิยมที่ใช้ในการวิเคราะห์และสร้างโมเดลเกี่ยวกับราคาหุ้น NVIDIA")   
st.subheader("1.ที่มาของ Dataset และ Feature")
st.markdown("""

ส่วนนี้จะอธิบายเกี่ยวกับ:
- Dataset ที่ใช้มาจาก:
  - [NVIDIA Common Stock Dataset](https://www.kaggle.com/datasets/bhargavchirumamilla/nvidia-corporation-common-stock-nvda-data/data)
- รายละเอียดของ Feature ที่ใช้:
  - Date: วันที่ของข้อมูล
  - Open: ราคาเปิดของหุ้น
  - High: ราคาสูงสุดในวันนั้น
  - Low: ราคาต่ำสุดในวันนั้น
  - Close/Last: ราคาปิดของหุ้น
  - Volume: ปริมาณการซื้อขาย
  - MA5: ค่าเฉลี่ยเคลื่อนที่ 5 วัน
  - MA20: ค่าเฉลี่ยเคลื่อนที่ 20 วัน
  - Price_Change: เปอร์เซ็นต์การเปลี่ยนแปลงของราคาเทียบกับวันก่อนหน้า

Dataset นี้เหมาะสำหรับการทำนายราคาหุ้นเนื่องจาก:
- มีข้อมูลราคาย้อนหลังที่ต่อเนื่อง
- มีปัจจัยทางเทคนิคที่สำคัญครบถ้วน
- สามารถคำนวณตัวชี้วัดเพิ่มเติมได้
- เป็นหุ้นที่มีสภาพคล่องสูง ทำให้ข้อมูลมีความน่าเชื่อถือ
""")

st.subheader("2. การอธิบายโมเดล Neural Network")
st.markdown("""
### 1. การเตรียมข้อมูล

#### การจัดเตรียมข้อมูล
- คำนวณ Price Change จากราคาปิดรายวัน
- ลบข้อมูลที่มีค่า NaN ออก
- เตรียม Features สำหรับโมเดล:
  - Open: ราคาเปิด
  - High: ราคาสูงสุด 
  - Low: ราคาต่ำสุด
  - Volume: ปริมาณการซื้อขาย
  - MA5: ค่าเฉลี่ยเคลื่อนที่ 5 วัน
  - MA20: ค่าเฉลี่ยเคลื่อนที่ 20 วัน
  - Price Change: การเปลี่ยนแปลงของราคา

#### การแบ่งข้อมูล
- แบ่งข้อมูลเป็นชุดฝึกสอน 80% และชุดทดสอบ 20%
- ปรับสเกลข้อมูลด้วย StandardScaler

### 2. ทฤษฎีของอัลกอริทึมที่ใช้

#### Linear Regression
- โมเดลพื้นฐานที่ใช้หาความสัมพันธ์เชิงเส้น
- เหมาะกับการทำนายค่าต่อเนื่อง
- ง่ายต่อการตีความผลลัพธ์

#### Random Forest
- อัลกอริทึมแบบ Ensemble ที่ใช้ต้นไม้ตัดสินใจหลายต้น
- ทนทานต่อ Outlier และข้อมูลรบกวน
- สามารถจัดการกับความสัมพันธ์แบบไม่เชิงเส้นได้ดี

#### Support Vector Regression (SVR)
- ใช้ kernel trick ในการแปลงข้อมูลไปยังมิติที่สูงขึ้น
- เหมาะกับข้อมูลที่มีความสัมพันธ์ซับซ้อน
- ใช้ RBF kernel ในการจัดการความไม่เป็นเชิงเส้น

### 3. ขั้นตอนการพัฒนาโมเดล

1. **การสร้างโมเดล**
   - สร้างโมเดลทั้ง 3 แบบ
   - กำหนดพารามิเตอร์เริ่มต้น เช่น n_estimators=100 สำหรับ Random Forest

2. **การฝึกสอนโมเดล**
   - ฝึกสอนแต่ละโมเดลด้วยข้อมูลชุดฝึกสอน
   - ทำนายผลลัพธ์ด้วยข้อมูลชุดทดสอบ

3. **การประเมินผล**
   - ใช้ metrics หลายตัวในการประเมิน:
     - RMSE (Root Mean Square Error)
     - R2 Score
   - เปรียบเทียบประสิทธิภาพระหว่างโมเดล
   - เลือกโมเดลที่ดีที่สุดตามค่า RMSE
""")
