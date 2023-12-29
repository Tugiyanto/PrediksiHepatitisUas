import pickle
import streamlit as st


model = pickle.load(open('model.sav', 'rb'))

st.title('Prediksi HepatitisC')

Age = st.number_input(
    'Masukkan umur pasien', min_value=19.0, max_value=77.0, step=1.0)

Sex = st.selectbox(
    'Masukkan jenis kelamin pasien.', ['Laki - Laki', 'Perempuan'])
if Sex == 'Laki - Laki':
    Sex = 0
else:
    Sex = 1

ALB = st.number_input(
    'Masukkan konsentrasi serum albumin pasien.', min_value=14.9, max_value=82.2, step=0.1)

ALP = st.number_input(
    'Masukkan nilai fosfatase alkali pasien.', min_value=11.3, max_value=416.6, step=0.1)

ALT = st.number_input(
    'Masukkan nilai enzim alanine transaminase pasien', min_value=0.90, max_value=325.30, step=0.1)

AST = st.number_input(
    'Masukkan nilai aspartate aminotransferase pasien.', min_value=10.6, max_value=324.0, step=0.1)

BIL = st.number_input(
    'Masukkan nilai bilirubin pasien.', min_value=0.80, max_value=254.0, step=0.1)

CHE = st.number_input(
    'Masukkan nilai kolinesterase pasien.', min_value=1.42, max_value=16.41, step=0.1)

CHOL = st.number_input(
    'Masukkan nilai kolesterol pasien.', min_value=1.43, max_value=9.67, step=0.1)

CREA = st.number_input(
    'Masukkan nilai kreatinin pasien.', min_value=8.0, max_value=1079.10, step=0.1)

GGT = st.number_input(
    'Masukkan nilai gamma-glutamyl transferase pasien.', min_value=4.50, max_value=650.90, step=0.1)

PROT = st.number_input(
    'Masukkan nilai protein total pasien.', min_value=44.80, max_value=90.0, step=0.1)

predict = ''

if st.button('Prediksi'):
    input_data = [[Age, Sex, ALB, ALP, ALT,
                   AST, BIL, CHE, CHOL, CREA, GGT, PROT]]
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        predict = 'Pasien diprediksi terkena Hepatitis.'
    elif prediction[0] == 1:
        predict = 'Pasien diprediksi tidak terkena Hepatitis.'
st.write(predict)