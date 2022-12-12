import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle

st.set_page_config(layout='centered', page_icon='💸', page_title='¿Aceptarías la oferta?')

st.title('Aplicación Realizada por Carlota Sánchez y Laura Rodríguez para ver si aceptarías la oferta de un banco para una tarjeta de crédito.')

st.image(Image.open('../images/banco.webp'))
st.sidebar.image(Image.open('../image/uax.jpeg'))
st.subheader('A continuación introduce los siguiente datos para que la aplicación pueda realizar la predicción:')

reward = float(st.text_input('Reward', 0))
mailer_type = float(st.text_input('Mailer Type', 0))
income = float(st.text_input('Income Level', 0))
accounts_open = float(st.text_input('# Bank Accounts Open', 0))
overdraft = float(st.text_input('Overdraft Protection', 0))
credit_rating = float(st.text_input('Credit Rating', 0))
cards_held = float(st.text_input('# Credit Cards Held', 0))
owns_home = float(st.text_input('# Homes Owned', 0))
size = float(st.text_input('Household Size', 0))
own_your_home = float(st.text_input('Own Your Home', 0))
balance = float(st.text_input('Average Balance', 0))

data = {'mean reward': reward,
        'mean mailer type': mailer_type,
        'mean income': income,
        'mean # bank accounts open': accounts_open,
        'mean overdraft protection': overdraft,
        'mean credit rating': credit_rating,
        'mean # credit cards held': cards_held,
        'mean # homes owned': owns_home,
        'mean househeld size': size,
        'mean own your home': own_your_home,
        'mean average balance': balance}

df = pd.DataFrame(data, index=[0])

st.subheader('Compruebe que los datos introducidos son correctos')

st.table(df)

enviar = st.button('Enviar datos')

if enviar:

    lr = pickle.load(open('../models/logisticregression.pkl', 'rb'))
    pred = lr.predict(df)

    if pred[0] == 1:
        st.title('''Buenas noticas
                    Con un 15.47 % de probabilidad podemos afirmar que va a aceptar la oferta''')
    else:
        st.title('''Malas noticias 
                    Con un 15.47 % de probabilidad podemos afirmar que va a rechazar la oferta''')