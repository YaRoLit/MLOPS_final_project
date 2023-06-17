import pandas as pd

import streamlit as st

import predict

import preproc as prc


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun

    return df.to_csv(sep='\t', encoding='utf-16')


st.set_page_config(page_title="Определение стоимости квартир(ы)",
                   page_icon="🏠")


st.write("# Приложение для определения стоимости квадратного метра жилплощади")

uploaded_df = st.file_uploader("Загрузите файл", help='Загрузите файл с объектами\
                               недвижимости в виде списка в формате txt, csv\
                               (разделитель tab), кодировка "utf-16")',
                               type=['txt', 'csv'])

if uploaded_df:
    uploaded_df = pd.read_csv(uploaded_df, sep='\t', encoding='utf-16')
    
    uploaded_df = prc.get_predict_model_features(uploaded_df)

    uploaded_df = prc.fill_nans(uploaded_df)

    uploaded_df = prc.create_adress_feature(uploaded_df)

    predict_df = predict.predict(uploaded_df)

    uploaded_df.insert(0, 'цена кв.м.', predict_df)

    st.dataframe(uploaded_df)

    csv = convert_df(uploaded_df)

    st.download_button(label="Загрузить CSV файл с расчётом цены",
                       data=csv,
                       file_name='price.csv',
                       mime='text/csv')
