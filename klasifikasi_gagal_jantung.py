import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tempfile import NamedTemporaryFile
import joblib
import os
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def unduh_hasil(df, prediksi, filename):
    hasil_df = df.copy()
    hasil_df['Prediksi'] = prediksi
    with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        hasil_df.to_excel(tmp.name, index=False)
        tmp_file = tmp.name

    with open(tmp_file, 'rb') as f:
        data = f.read()
    st.download_button(
        label="Unduh Hasil Klasifikasi",
        data=data,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    os.remove(tmp_file)

def main():
    st.title('Klasifikasi Penyakit Gagal Jantung dengan Naive Bayes')

    st.sidebar.header('Unggah Data Training')
    uploaded_train_file = st.sidebar.file_uploader("Unggah file Excel untuk data training", type=["xlsx", "xls"])

    if uploaded_train_file is not None:
        # Baca data training
        df_train = pd.read_excel(uploaded_train_file)

        st.subheader('Data Training:')
        st.write(df_train)

        # Pisahkan fitur (X) dan target (y)
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]

        # Identifikasi kolom numerik dan kategori
        numeric_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
        categorical_features = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

        # Proses preprocessing untuk data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Buat pipeline dengan preprocessing dan model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())])

        # Latih model dan sediakan tautan unduh untuk model
        if st.sidebar.button('Latih Model dan Unduh'):
            pipeline.fit(X_train, y_train)

            # Simpan model ke file sementara
            with NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                joblib.dump(pipeline, tmp.name)
                tmp_file = tmp.name

            # Buat tautan unduh untuk model
            with open(tmp_file, 'rb') as f:
                data = f.read()
            st.download_button(
                label="Unduh Model",
                data=data,
                file_name='trained_model.pkl',
                mime='application/octet-stream'
            )
            os.remove(tmp_file)

    # Sidebar untuk mengunggah data testing dan model untuk prediksi
    st.sidebar.header('Unggah Data Testing dan Model')
    uploaded_test_file = st.sidebar.file_uploader("Unggah file Excel untuk data testing", type=["xlsx", "xls"])
    uploaded_model_file = st.sidebar.file_uploader("Unggah file model yang telah diunduh (pkl)", type=["pkl"])

    if uploaded_test_file is not None and uploaded_model_file is not None:
        # Baca data testing
        df_test = pd.read_excel(uploaded_test_file)

        st.subheader('Data Testing:')
        st.write(df_test)

        # Muat model yang diunggah
        model = joblib.load(uploaded_model_file)

        # Pastikan X_test adalah DataFrame
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        # Prediksi menggunakan data testing
        y_pred = model.predict(X_test)

        # Hitung akurasi
        akurasi = accuracy_score(y_test, y_pred)

        # Tampilkan hasil evaluasi
        st.subheader('Hasil Evaluasi:')
        st.write(f'Akurasi: {akurasi:.2f}')
       
        # Tampilkan Confusion Matrix dengan label
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Aktual Negatif', 'Aktual Positif'], columns=['Prediksi Negatif', 'Prediksi Positif'])
        cm_df.index.name = 'Aktual'
        cm_df.columns.name = 'Prediksi'

        # Menambahkan label TP, TN, FP, FN
        cm_labels = pd.DataFrame([['TN', 'FP'], ['FN', 'TP']], index=['Aktual Negatif', 'Aktual Positif'], columns=['Prediksi Negatif', 'Prediksi Positif'])
        cm_annotated = cm_df.astype(str) + '\n' + cm_labels

        st.write('Confusion Matrix:')
        st.write(cm_annotated)       
        
        # Hasilkan laporan klasifikasi
        laporan = classification_report(y_test, y_pred, output_dict=True)
        laporan_df = pd.DataFrame(laporan).transpose()
        laporan_tabel = tabulate(laporan_df, headers='keys', tablefmt='pipe', numalign='right')
        st.text(laporan_tabel)

        # Unduh hasil klasifikasi untuk data testing
        st.subheader('Unduh Hasil Klasifikasi (Data Testing):')
        unduh_hasil(df_test, y_pred, 'predicted_results_test.xlsx')

    # Sidebar untuk input data baru secara manual
    st.sidebar.header('Input Data Baru')
    fitur = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']
    input_data = {}

    input_data['age'] = st.sidebar.number_input('Masukkan Umur', min_value=0)
    input_data['sex'] = st.sidebar.selectbox('Pilih Jenis Kelamin', options=[0, 1], format_func=lambda x: 'Laki-laki' if x == 0 else 'Perempuan')
    input_data['chest pain type'] = st.sidebar.selectbox('Pilih Tipe Nyeri Dada', options=[1, 2, 3, 4], format_func=lambda x: {
        1: 'Typical Angina (Angina Tipikal): Nyeri sering digambarkan sebagai rasa tertekan atau nyeri di belakang tulang dada yang dapat menjalar ke lengan, leher, rahang, punggung, atau perut.',
        2: 'Atypical Angina (Angina Atipikal): Terasa lebih ringan atau tidak seperti rasa tertekan, mungkin lebih seperti rasa terbakar, kram, atau nyeri tumpul.',
        3: 'Non-Anginal Pain (Nyeri Non-Angina): Nyeri yang disebabkan oleh kondisi lain seperti masalah pencernaan, otot, tulang, atau paru-paru.',
        4: 'Asymptomatic (Asimtomatik): Tidak ada gejala nyeri dada yang terasa meskipun mungkin ada masalah jantung yang mendasarinya.'
    }[x])
    input_data['resting bp s'] = st.sidebar.number_input('Masukkan Tekanan Darah Istirahat', min_value=0)
    input_data['cholesterol'] = st.sidebar.number_input('Masukkan Kolesterol', min_value=0)
    input_data['fasting blood sugar'] = st.sidebar.selectbox('Gula Darah Puasa > 120', options=[0, 1], format_func=lambda x: 'Salah' if x == 0 else 'Benar')
    input_data['resting ecg'] = st.sidebar.selectbox('Hasil Elektrodigrafi', options=[0, 1], format_func=lambda x: 'Normal' if x == 0 else 'Tidak Normal')
    input_data['max heart rate'] = st.sidebar.number_input('Masukkan Detak Jantung Maksimum', min_value=0)
    input_data['exercise angina'] = st.sidebar.selectbox('Nyeri Dada Apabila Olahraga', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Iya')
    input_data['oldpeak'] = st.sidebar.number_input('Masukkan Oldpeak', min_value=0.0, format="%.2f")
    input_data['ST slope'] = st.sidebar.selectbox('Pilih ST Slope', options=[1, 2, 3], format_func=lambda x: 'Upsloping' if x == 1 else 'Flat' if x == 2 else 'Downsloping')
    
    if st.sidebar.button('Klasifikasi Data Baru'):
        if uploaded_model_file is not None:
            model = joblib.load(uploaded_model_file)
            input_df = pd.DataFrame([input_data])
            # Pastikan kolom DataFrame sama dengan yang dilatih
            input_df = input_df[fitur]
            # Proses input_df dengan preprocessor
            input_values = input_df.values
            klasifikasi = model.predict(input_df)
            hasil_klasifikasi = 'Tidak Risiko' if klasifikasi[0] == 0 else 'Risiko'
            st.subheader('Hasil Klasifikasi Data Baru:')
            st.write(f'Klasifikasi: {hasil_klasifikasi}')
        else:
            st.sidebar.error('Silakan unggah model terlebih dahulu.')

if __name__ == '__main__':
    main()
