#import libraries
from sklearn import model_selection
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import joblib


#load in pipeline
model_file = 'final_model.sav'
model = pickle.load(open(model_file,'rb'))

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

def main():
    
    # get data and convert data into dataframe
    with st.form(key='my_form'):
        Location = st.selectbox('LOCATION:', ['Lekki','Ajah','Ikoyi','Ibeju','Ikorodu','Victoria Island','Ipaja','Amuwo-Odofin','Isolo','Ikeja','Ojodu',                  
                    'Yaba','Magodo','Alimosho','Surulere','Shomolu','Ikotun/Igando','Abule Egba','Gbagada','Ogudu','Ogba','Ifako-Ijaiye','Epe','Agege',                  
                    'Kosofe','Maryland','Ilupeju','Agboyi/Ketu','Mushin','Ojo','Oshodi','Ilashe','Badagry','Agbara-Igbesan','Ojota','Orile','Lagos Island (Eko)',      
                    'Ejigbo','Lagos State','Apapa','Eko Atlantic'])
        House_type = st.selectbox('HOUSE_TYPES',['FLAT','DUPLEX', 'BUNGALOW', 'HOUSE'])
        Number_of_bedrooms = st.number_input('BEDROOMS', min_value=1, max_value=9, value=1)
        Number_of_bathrooms = st.number_input('BATHROOMS', min_value=1, max_value=9, value=1)
        Number_of_toilets = st.number_input('TOILETS', min_value=1, max_value=9, value=1)
        House_Condition = st.selectbox('HOUSE_CONDITION',['NEW', 'OLD'])
        Furnishing = st.selectbox('FURNISHED',['YES', 'NO'])
        Servicing = st.selectbox('SERVICED',['YES', 'NO'])
        
        submit_button = st.form_submit_button(label='Predict Price') 
        
         #define dataframe
    
    data_df = pd.DataFrame([[Location, House_type,Number_of_bedrooms,Number_of_bathrooms,Number_of_toilets,House_Condition,Furnishing,Servicing]],
                           columns = ['LOCATION','HOUSE_TYPES', 'BEDROOMS', 'BATHROOMS', 'TOILETS', 'HOUSE_CONDITION', 'FURNISHED','SERVICED'])

    #prediction
    price = model.predict(data_df)
    
    if submit_button:
        st.write('The predicted price of the property is: ₦{:,.2f}'.format(price[0]))
main()   
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
