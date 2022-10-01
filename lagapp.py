import xgboost as xgb
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import joblib

xgb_model = joblib.load('final_xgboost_model.pkl')

st.set_page_config(
    layout='wide',
    page_title='Lagos House Rent Prediction',
    page_icon='img.jfif',
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: shown;}
            footer {visibility: hidden;}
            footer:after {
                          content:'Created by TrailBlazersNG Team'; 
                          visibility: visible;
                          display: block;
                          position: relative;
                          #background-color: white;
                          padding: 4px;
                          top: 2px;
                          }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def set_background(png_file):
    bin_str = (png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: black;'> Lagos House Rent Prediction</h1>", unsafe_allow_html=True)
image = Image.open('image3.jpg')
st.image(image, caption='PHOTO FROM ROYALTY-FREE STOCK PHOTO')

Location = st.selectbox('House Location:', ['Ikoyi', 'Yaba', 'Lekki', 'Ajah', 'Victoria island', 'Ikeja',
                                            'Ilupeju', 'Isolo', 'Shomolu', 'Ketu', 'Surulere', 'amuwo odofin',
                                            'abule egba', 'Oshodi', 'Apapa', 'ajah', 'ikorodu', 'ojodu',
                                            'ipaja', 'Egbeda', 'Ikotun', 'idimu', 'Ogba', 'igando', 'akowonjo',
                                            'ikate-lekki', 'ikota-lekki', 'Chevron-lekki', 'phase1-lekki',
                                            'Phase2-lekki', 'Vgc-lekki', 'ibeju-lekki', 'osapa london lekki',
                                            'Agungi lekki', 'lekki', 'victoria island', 'opebi-ikeja',
                                            'Allen avenue-ikeja', 'gra-ikeja', 'Oregun-ikeja', 'ikeja',
                                            'akoka-yaba', 'Alagomeji-yaba', 'adekunle-yaba', 'abule oja-yaba',
                                            'abule ijesha-yaba', 'Onike-yaba', 'jibowu-yaba', 'Sabo-yaba',
                                            'iwaya-yaba', 'ebute metta-yaba', 'Fola agoro-yaba',
                                            'ago palace-okota', 'Okota', 'phase1-gbagada', 'Phase2-gbagada',
                                            'ifako-gbagada', 'Oworonshoki-gbagada', 'Soluyi-gbagada',
                                            'medina-gbagada', 'gbagada', 'anthony village-maryland',
                                            'mende-maryland', 'maryland', 'Ikotun-igando', 'ojo', 'Ayobo',
                                            'Akesan', 'egbeda', 'fagba-agege', 'Cement-agege', 'Oko oba-agege',
                                            'ifako-agege', 'iju ishaga-agege', 'orile-agege', 'agege',
                                            'Marina-lagos island', 'Obalende-lagos island', 'Lagos island',
                                            'Ilasamaja-mushin', 'Mushin', 'Phase1-magodo', 'Phase2-magodo',
                                            'Ojota', 'Ogudu', 'Alapere-ketu', 'Mile12-ketu', 'Shangisha-ketu',
                                            'ikosi-ketu', 'Epe', 'Alimosho', 'Bariga', 'ejigbo', 'Sangotedo',
                                            'Badagry', 'Iganmu-orile', 'Fagba-agege', 'Iju ishaga-agege'])

House_type = st.selectbox('Type of House:', ["FLAT", "HOUSE", "DUPLEX", "BUNGALOW"])
Number_of_bedrooms = st.number_input('Number of bedroom:', min_value=1, max_value=9, value=1)
Number_of_toilets = st.number_input('Number of toilet:', min_value=1, max_value=9, value=1)
Number_of_bathrooms = st.number_input('Number of bathrooms:', min_value=1, max_value=9, value=1)


def predict(Number_of_bedrooms, Number_of_bathrooms, Number_of_toilets, Location, House_type):
    new_location = {'Ikoyi': 0, 'Yaba': 1, 'Lekki': 2, 'Ajah': 3, 'Victoria island': 4, 'Ikeja': 5, 'Ilupeju': 6,
                    'Isolo': 7,
                    'Shomolu': 8, 'Ketu': 9, 'Surulere': 10, 'amuwo odofin': 11, 'abule egba': 12, 'Oshodi': 13,
                    'Apapa': 14,
                    'ajah': 15, 'ikorodu': 16, 'ojodu': 17, 'ipaja': 18, 'Egbeda': 19, 'Ikotun': 20, 'idimu': 21,
                    'Ogba': 22,
                    'igando': 23, 'akowonjo': 24, 'ikate-lekki': 25, 'ikota-lekki': 26, 'Chevron-lekki': 27,
                    'phase1-lekki': 28,
                    'Phase2-lekki': 29, 'Vgc-lekki': 30, 'ibeju-lekki': 31, 'osapa london lekki': 32,
                    'Agungi lekki': 33,
                    'lekki': 34, 'victoria island': 35, 'opebi-ikeja': 36, 'Allen avenue-ikeja': 37, 'gra-ikeja': 38,
                    'Oregun-ikeja': 39, 'ikeja': 40, 'akoka-yaba': 41, 'Alagomeji-yaba': 42, 'adekunle-yaba': 43,
                    'abule oja-yaba': 44, 'abule ijesha-yaba': 45, 'Onike-yaba': 46, 'jibowu-yaba': 47, 'Sabo-yaba': 48,
                    'iwaya-yaba': 49, 'ebute metta-yaba': 50, 'Fola agoro-yaba': 51, 'ago palace-okota': 52,
                    'Okota': 53,
                    'phase1-gbagada': 54, 'Phase2-gbagada': 55, 'ifako-gbagada': 56, 'Oworonshoki-gbagada': 57,
                    'Soluyi-gbagada': 58, 'medina-gbagada': 59, 'gbagada': 60, 'anthony village-maryland': 61,
                    'mende-maryland': 62, 'maryland': 63, 'Ikotun-igando': 64, 'ojo': 65, 'Ayobo': 66, 'Akesan': 67,
                    'egbeda': 68,
                    'fagba-agege': 69, 'Cement-agege': 70, 'Oko oba-agege': 71, 'ifako-agege': 72,
                    'iju ishaga-agege': 73,
                    'orile-agege': 74, 'agege': 75, 'Marina-lagos island': 76, 'Obalende-lagos island': 77,
                    'Lagos island': 78,
                    'Ilasamaja-mushin': 79, 'Mushin': 80, 'Phase1-magodo': 81, 'Phase2-magodo': 82, 'Ojota': 83,
                    'Ogudu': 84,
                    'Alapere-ketu': 85, 'Mile12-ketu': 86, 'Shangisha-ketu': 87, 'ikosi-ketu': 88, 'Epe': 89,
                    'Alimosho': 90,
                    'Bariga': 91, 'ejigbo': 92, 'Sangotedo': 93, 'Badagry': 94, 'Iganmu-orile': 95, 'Fagba-agege': 96,
                    'Iju ishaga-agege': 97}

    for i in new_location:
        if Location == i:
            Location = new_location[i]

    if House_type == 'FLAT':
        House_type = 1
    elif House_type == 'HOUSE':
        House_type = 2
    elif House_type == 'DUPLEX':
        House_type = 3
    elif House_type == 'BUNGALOW':
        House_type = 4

    prediction = xgb_model.predict(
        pd.DataFrame([[Number_of_bedrooms, Number_of_bathrooms, Number_of_toilets, Location, House_type]],
                     columns=['BEDROOMS', 'BATHROOMS', 'TOILETS', 'LOCATION', 'HOUSE_TYPE']))
    return prediction
if st.button('Predict Price'):
    price = predict(Number_of_bedrooms, Number_of_bathrooms, Number_of_toilets, Location, House_type)
    st.write('The predicted price of the property is: ₦{:,.2f}'.format(price[0]))

st.title('Contributors')
st.text("""
        This app was developed by:
        """)
st.write('- Michael Onabanjo (https://github.com/Onabanjomicheal)')
st.write('- Babajide Alao (https://github.com/BabajideAlao-knn)')
st.write('- Paul Adegbite (https://github.com/octopuspaul110)')
st.write('- Alinta Innocent (https://github.com/aliNtainnocent)')
st.write('- Kehinde Olalekan (https://github.com/KENNYDGREAT2)')

st.info('During #DSRoom Project challenge under the mentorship of Samson Afolabi (https://twitter.com/samsonafo)')
