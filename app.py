import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
import plotly.express as px

# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen SpotifyWrapped 2024")

    # Bagian untuk upload file
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        # Membaca file CSV
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model_smote.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        if model and vectorizer:
            # Pastikan kolom 'stemming_data' ada dalam file
            if 'stemming_data' in data.columns:
                # Validasi dan pastikan kolom 'stemming_data' berupa string
                data['stemming_data'] = data['stemming_data'].astype(str)

                # Prediksi sentimen
                if st.button("Prediksi Sentimen"):
                    try:
                        # Transformasi data menggunakan vectorizer
                        X = vectorizer.transform(data['stemming_data'])
                        predictions = model.predict(X)

                        # Menambahkan hasil prediksi ke dalam data
                        data['Predicted Sentiment'] = predictions
                        st.write("Hasil Prediksi Sentimen:")
                        st.write(data[['stemming_data', 'Predicted Sentiment']])

                        # Visualisasi distribusi hasil sentimen
                        sentiment_counts = data['Predicted Sentiment'].value_counts()
                        st.write("Distribusi Hasil Sentimen:")

                        # Membuat Bar Chart untuk distribusi sentimen
                        fig_bar = px.bar(
                            sentiment_counts,
                            x=sentiment_counts.index,
                            y=sentiment_counts.values,
                            labels={'x': 'Sentimen', 'y': 'Jumlah'},
                            title="Distribusi Sentimen"
                        )
                        st.plotly_chart(fig_bar)

                        # Tombol untuk mengunduh hasil prediksi
                        st.download_button(
                            label="Download Hasil Prediksi",
                            data=data.to_csv(index=False),
                            file_name="hasil_prediksi.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Terjadi error saat prediksi: {e}")
            else:
                st.error("Kolom 'stemming_data' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
