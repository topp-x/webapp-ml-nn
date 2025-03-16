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
st.header("Topx")

st.subheader("เว็บแอพพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อสาธิตการทำงานของ:")
st.markdown("""
- Machine Learning (ML)
- Neural Network (NN)

### 📚 เนื้อหา
1. **คำอธิบาย Machine Learning**: ทำความเข้าใจพื้นฐานและหลักการทำงานของ ML
2. **คำอธิบาย Neural Network**: เรียนรู้เกี่ยวกับโครงสร้างและการทำงานของ NN
3. **ทดลอง Machine Learning**: ทดลองการทำงานของโมเดล ML
4. **ทดลอง Neural Network**: ทดลองการทำงานของโมเดล NN
""")
