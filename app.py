import streamlit as st
import pandas as pd
import pickle

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen dengan Model ML")

    # Gaya tambahan untuk aplikasi
    html_temp = """
    <div style="background-color:cyan;padding:10px">
    <h1 style="color:black;text-align:center;">Streamlit Sentiment Analysis App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Bagian untuk upload file
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data.head())

        # Load model dan vectorizer
        with open('rf_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        # Pastikan kolom 'stemming_data' ada dalam file
        if 'stemming_data' in data.columns:
            # Prediksi sentimen
            if st.button("Prediksi Sentimen"):
                X = vectorizer.transform(data['stemming_data'])
                predictions = model.predict(X)
                
                # Menambahkan hasil prediksi ke data
                data['Predicted Sentiment'] = predictions
                st.write("Hasil Prediksi:")
                st.write(data[['stemming_data', 'Predicted Sentiment']].head())

                # Tombol untuk mengunduh hasil
                st.download_button(
                    label="Download Hasil Prediksi",
                    data=data.to_csv(index=False),
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
        else:
            st.error("Kolom 'stemming_data' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
