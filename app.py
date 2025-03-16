import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
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

st.title("Web Application for Machine Learning and Neural Network")
st.write("Dev by Topx")

# st.subheader("เว็บแอพพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อทดลองการทำงานของโมเดล Machine Learning และ Neural Network")
# st.markdown("""
# - Machine Learning (ML)
# - Neural Network (NN)
# """)
            
st.markdown("""
### 👋 Welcome to my website !

เว็บไซต์นี้จะพาคุณไปเรียนรู้และทำความเข้าใจเกี่ยวกับ Machine Learning และ Neural Network ผ่านตัวอย่างที่เข้าใจง่าย พร้อมการสาธิตการใช้งานจริง

#### 🎯 สิ่งที่คุณจะได้เรียนรู้:
- เข้าใจหลักการทำงานของ Machine Learning ผ่านการทำนายโรคหัวใจ
- เรียนรู้การทำงานของ Neural Network ผ่านการวิเคราะห์ราคาหุ้น NVIDIA
- ทดลองใช้โมเดลจริงด้วยตัวคุณเอง
- เห็นการประยุกต์ใช้งานในสถานการณ์จริง

#### 💡 เหมาะสำหรับ:
- ผู้ที่สนใจด้าน AI และ Data Science
- นักศึกษาและผู้เริ่มต้นเรียนรู้
- ผู้ที่ต้องการเห็นการประยุกต์ใช้งานจริง

#### 🚀 เริ่มต้นได้ง่ายๆ:
1. เลือกหัวข้อที่สนใจจากเมนูด้านบน
2. อ่านคำอธิบายและทำความเข้าใจ
3. ลองทดสอบโมเดลด้วยตัวคุณเอง
""")

st.markdown("---")

# st.markdown("""
# 📚 เนื้อหา
# 1. **คำอธิบาย Machine Learning**: ทำความเข้าใจพื้นฐานและหลักการทำงานของ ML
# 2. **คำอธิบาย Neural Network**: เรียนรู้เกี่ยวกับโครงสร้างและการทำงานของ NN
# 3. **ทดลอง Machine Learning**: ทดลองการทำงานของโมเดล ML
# 4. **ทดลอง Neural Network**: ทดลองการทำงานของโมเดล NN
# """)
