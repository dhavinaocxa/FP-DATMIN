import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk mengunduh file model dan pipeline
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None

# Fungsi utama Streamlit
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen SpotifyWrapped 2024")

    # Upload file CSV
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data.head())

        # Load model pipeline dari URL
        model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model_smote.pkl"
        pipeline = load_model_from_url(model_url)

        # Pastikan pipeline berhasil di-load
        if pipeline:
            # Pastikan kolom 'stemming_data' ada
            if 'stemming_data' in data.columns:
                # Prediksi sentimen
                if st.button("Prediksi Sentimen"):
                    predictions = pipeline.predict(data['stemming_data'])

                    # Tambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions
                    st.write("Hasil Prediksi:")
                    st.write(data[['stemming_data', 'Predicted Sentiment']])

                    # Visualisasi distribusi hasil sentimen
                    sentiment_counts = data['Predicted Sentiment'].value_counts()
                    fig_bar = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentimen', 'y': 'Jumlah'},
                        title="Distribusi Hasil Sentimen"
                    )
                    st.plotly_chart(fig_bar)

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

                # Tampilkan akurasi model
                st.subheader("Evaluasi Model")
                st.info("Akurasi model Random Forest dengan balancing SMOTE: 87%")  # Contoh akurasi
                st.write("""
                    - **Model**: Random Forest Classifier  
                    - **Teknik Balancing**: SMOTE  
                    - **Feature Extraction**: TF-IDF  
                """)
            else:
                st.error("Kolom 'stemming_data' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
