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
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        if model and vectorizer:
            # Pastikan kolom 'stemming_data' ada dalam file
            if 'stemming_data' in data.columns:
                # Prediksi sentimen
                if st.button("Prediksi Sentimen"):
                    X = vectorizer.transform(data['stemming_data'])
                    predictions = model.predict(X)

                    # Menambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions
                    st.write("Hasil Prediksi:")
                    st.write(data[['stemming_data', 'Predicted Sentiment']])

                    # Visualisasi distribusi hasil sentimen
                    sentiment_counts = data['Predicted Sentiment'].value_counts()

                    # Membuat Bar Chart untuk distribusi sentimen
                    fig_bar = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentimen', 'y': 'Jumlah'},
                        title="Distribusi Hasil Sentimen"
                    )
                    st.plotly_chart(fig_bar)

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
