import joblib
import streamlit as st

model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar Publik")
st.write("Aplikasi ini dibuat menggunakan Teknologi NLP dengan memanfaatkan model machine learning Logistic Regression")
input = st.text_input("Masukkan Komentar Anda!!")
if st.button("Submit"):
    if input.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]
        
        label_map = {
            0: "Negatif",
            1: "Positif"
            
        }

        st.subheader("Hasil Analisis Komentar")
        st.write("**Komentar :**", label_map.get(prediksi, prediksi))
