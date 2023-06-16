import pandas as pd

import streamlit as st

import predict


st.set_page_config(page_title="Определение стоимости квартир(ы)",
                   page_icon="🏠")


st.write("# Приложение для определения стоимости квадратного метра жилплощади")

col1, col2 = st.columns(2)

with col1:
    adress = st.text_input('Введите адрес')

    part = st.selectbox('Выберите район города', ('САО', 'ЛАО', 'КАО', 'ЦАО', 'ОАО'))

    material = st.selectbox('Выберите материал стен', ('Кирпичные',
                                                       'Крупнопанельные',
                                                       'Каменные',
                                                       'Крупноблочные',
                                                       'Деревянные',
                                                       'Блочные',
                                                       'Шлакобетонные',
                                                       'Бетонные',
                                                       'Рубленые',
                                                       'Каркасно-засыпные'))

    build_year = st.number_input('Введите год постройки', min_value=1900)

with col2:
    square = st.number_input('Введите площадь помещения', min_value=10)

    floor = st.number_input('Введите этаж', min_value=1)

    floor_range = st.number_input('Введите этажность здания', min_value=1, max_value=24)

    rooms = st.number_input('Введите количество комнат', min_value=1)


btn = st.button("Рассчитать цену",
                help="Жмакай кнопку только после выбора файла")


if btn:
    test = pd.DataFrame({'год постройки': build_year,
                     'площадь': square,
                     'этаж помещения': floor,
                     'этажность здания': floor_range,
                     'комнат_1_2_3': rooms,
                     'Расст_до_магазинов': 2950,
                     'Расст_до_исторического центра_центр': 3500,
                     'материал': material,
                     'этажность помещения (группа)': 'средний',
                     'округ': part,
                     'адрес': adress}, index=[0])

    st.write("Стоимость кв.м. квартиры в данном жилье составляет:", float(predict.predict(test)))

