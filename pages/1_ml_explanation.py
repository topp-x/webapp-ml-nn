import streamlit as st

st.set_page_config(
    page_title="Machine Learning Explanation",
    page_icon="📚",
    layout="wide"
)


# สร้างเมนูนำทาง
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

st.header("การอธิบายโมเดล Machine Learning")
st.write("ไฟล์นี้เป็นชุดข้อมูลเกี่ยวกับโรคหัวใจ โดยดูจากปัจจัยต่าง ๆ ที่อาจมีผลต่อโรคหัวใจจากแหล่งข้อมูล Cleveland Heart Disease dataset ซึ่งเป็นหนึ่งในชุดข้อมูลยอดนิยมที่ใช้ในการวิเคราะห์และสร้างโมเดลเกี่ยวกับโรคหัวใจ")
st.subheader("1.ที่มาของ Dataset และ Feature")
st.markdown("""
ส่วนนี้จะอธิบายเกี่ยวกับ:
- Dataset ที่ใช้มาจาก:
  - [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/alihamza01/heart-disease-uci)
- รายละเอียดของ Feature ที่ใช้:
  - age: อายุของผู้ป่วย (ปี)
  - sex: เพศของผู้ป่วย (1 = ชาย, 0 = หญิง)
  - cp: ประเภทของอาการเจ็บหน้าอก
    - 0: อาการเจ็บหน้าอกปกติ
    - 1: อาการเจ็บหน้าอกผิดปกติ
    - 2: อาการเจ็บที่ไม่เกี่ยวกับหัวใจ
    - 3: ไม่มีอาการ
  - trestbps: ความดันโลหิตขณะพัก (มม.ปรอท)
  - chol: คอเลสเตอรอลในเลือด (มก./ดล.)
  - fbs: น้ำตาลในเลือดขณะอดอาหาร > 120 มก./ดล. (1 = จริง, 0 = เท็จ)
  - restecg: ผลคลื่นไฟฟ้าหัวใจขณะพัก
    - 0: ปกติ
    - 1: มีความผิดปกติของคลื่น ST-T
    - 2: แสดงการขยายตัวของหัวใจห้องซ้าย
  - thalch: อัตราการเต้นของหัวใจสูงสุด
  - exang: อาการเจ็บหน้าอกจากการออกกำลังกาย (1 = มี, 0 = ไม่มี)
  - oldpeak: การเปลี่ยนแปลงของคลื่น ST จากการออกกำลังกาย
  - slope: ความชันของส่วนสูงสุดของคลื่น ST
    - 0: ขึ้น
    - 1: ราบ
    - 2: ลง
  - ca: จำนวนหลอดเลือดหัวใจที่มีการตีบ (0-3)
  - thal: ผลการตรวจการไหลเวียนเลือดที่หัวใจ
    - 3: ปกติ
    - 6: ตำแหน่งที่มีการบกพร่อง
    - 7: ตำแหน่งที่มีการบกพร่องที่สามารถแก้ไขได้
  - target: การวินิจฉัยโรคหัวใจ (1 = มีโรค, 0 = ไม่มีโรค)

Dataset นี้เหมาะสำหรับการทำนายการเกิดโรคหัวใจเนื่องจาก:
- มีตัวแปรที่เกี่ยวข้องกับโรคหัวใจโดยตรง
- มีตัวแปรเป้าหมาย (Target Variable) ชัดเจน
- เป็นชุดข้อมูลที่ผ่านการศึกษาและใช้งานอย่างแพร่หลาย        
""")


# st.subheader("2. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
#### การเตรียมข้อมูล
- **การจัดการข้อมูลที่หายไป (Missing Values)**
  - ใช้ KNNImputer สำหรับข้อมูลตัวเลข โดยใช้ค่าเฉลี่ยจาก 5 เพื่อนบ้านที่ใกล้ที่สุด
  - ใช้ SimpleImputer สำหรับข้อมูลประเภท โดยใช้ค่าที่พบบ่อยที่สุด

- **การสร้างคุณลักษณะใหม่ (Feature Engineering)**
  - สร้างกลุ่มอายุ (age_group) แบ่งเป็น 5 ช่วง
  - สร้างตัวแปรบ่งชี้ผู้สูงอายุ (is_elderly)
  - จัดกลุ่มความดันโลหิต (bp_category)
  - คำนวณ Heart Rate Reserve
  - จัดกลุ่มอัตราการเต้นหัวใจ (hr_category)
  - จัดกลุ่มคอเลสเตอรอล (chol_category)
  - สร้างตัวแปรปฏิสัมพันธ์ระหว่างอายุ ความดัน และอัตราการเต้นหัวใจ

#### 2. ทฤษฎีของอัลกอริทึมที่ใช้

**1. Logistic Regression**
- เป็นอัลกอริทึมการเรียนรู้แบบมีผู้สอนสำหรับการจำแนกประเภท
- ใช้ฟังก์ชัน Sigmoid เพื่อทำนายความน่าจะเป็นของคลาส
- เหมาะกับการจำแนกประเภทแบบ 2 คลาส

**2. Random Forest**
- เป็นอัลกอริทึมแบบ Ensemble ที่สร้างต้นไม้ตัดสินใจหลายต้น
- แต่ละต้นถูกสร้างจากข้อมูลและคุณลักษณะที่สุ่มเลือก
- ผลลัพธ์สุดท้ายได้จากการโหวตเสียงส่วนใหญ่

**3. Histogram Gradient Boosting**
- เป็นอัลกอริทึมแบบ Boosting ที่สร้างโมเดลเป็นลำดับ
- แต่ละโมเดลพยายามแก้ไขข้อผิดพลาดของโมเดลก่อนหน้า
- ใช้ histogram เพื่อเพิ่มประสิทธิภาพการคำนวณ

#### 3. ขั้นตอนการพัฒนาโมเดล

1. **การแบ่งข้อมูล**
   - แบ่งข้อมูลเป็นชุดฝึกสอน 80% และชุดทดสอบ 20%

2. **การปรับมาตรฐานข้อมูล**
   - ใช้ StandardScaler เพื่อปรับข้อมูลให้มีค่าเฉลี่ย = 0 และส่วนเบี่ยงเบนมาตรฐาน = 1

3. **การจัดการข้อมูลไม่สมดุล**
   - ใช้เทคนิค SMOTE เพื่อสร้างข้อมูลสังเคราะห์สำหรับคลาสที่มีจำนวนน้อย

4. **การฝึกโมเดล**
   - ฝึกโมเดลทั้ง 3 แบบ
   - ใช้ class_weight='balanced' เพื่อให้น้ำหนักกับคลาสที่มีจำนวนน้อย

5. **การปรับแต่งพารามิเตอร์**
   - ใช้ GridSearchCV เพื่อหาพารามิเตอร์ที่ดีที่สุดสำหรับ Random Forest
   - พารามิเตอร์ที่ปรับแต่ง: max_depth, n_estimators, min_samples_split

6. **การประเมินผล**
   - ใช้ accuracy score และ classification report
   - วิเคราะห์ precision, recall, และ f1-score ของแต่ละคลาส
""")
