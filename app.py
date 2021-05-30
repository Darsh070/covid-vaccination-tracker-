#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import os
import urllib.request
import requests
from bs4 import BeautifulSoup
import camelot.io as camelot
import ghostscript
import html5lib
import json
from PIL import Image
import plotly.express as px
import streamlit as st
import folium
import plotly.io as pio
pio.renderers.default = 'browser'


# In[12]:



url = 'http://www.mohfw.gov.in/'
r=requests.get(url)
thepage = urllib.request.urlopen(url)
soup = BeautifulSoup(thepage)
for i in soup.find_all('div',attrs={'class':'col-sm-2 btns'}):
    pdf = i.a['href']    


# In[13]:


s_code = pd.read_csv('s_code.csv')


# In[14]:


gsheetid_1 = "1noqoXm0pnb61miW0HnaCCNsKPm_4lwhbzmHkREuF0ZY"
sheet_name_1 = "Vaccine"
gsheet_url_1 = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid_1,sheet_name_1)

data3 = pd.read_csv(gsheet_url_1)


# In[15]:


data3 = data3[data3["State"]!="India"]
data3.head()
data3 = data3.groupby('State').last().reset_index()
data3 = data3.drop(columns=['Updated On'])
data3['id']=s_code['id']


# In[16]:


states = json.load(open("states_india.geojson", "r"))
state_id_map = {}
for feature in states["features"]:
    feature["id"] = feature["properties"]["state_code"]
    state_id_map[feature["properties"]["st_nm"]] = feature["id"]


# In[17]:


st.markdown("# COVID-19 Vaccination Detailed Analysis (India)") 
img = Image.open('C:/Users/Darshit/shield.jpg')
if img.mode != 'RGB':
    img = img.convert('RGB')
st.image(img, caption = "Source: gislounge.com", width=700)
if st.checkbox('view_data'):
    st.subheader('Vaccination Data')
    st.write(data3)


# In[18]:


st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore our app")

st.sidebar.subheader('Visualizations')
if st.sidebar.checkbox('Individuals Vaccinated'):
    st.subheader('Number of Individuals Vaccinated by State')
    fig = px.bar(data3, x='State', y="Total Individuals Vaccinated", height=800, width = 800)
    st.plotly_chart(fig)

if st.sidebar.checkbox('Covid-19 Vaccines in India'):
    st.subheader('Vaccines Administered by State')
    fig = px.bar(data3, x="State", y=["Total Covaxin Administered","Total CoviShield Administered"], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('Gender'):
    st.subheader('Vaccines Administered by State and Gender')
    fig = px.bar(data3, x="State", y=["Male(Individuals Vaccinated)","Female(Individuals Vaccinated)",
                                     "Transgender(Individuals Vaccinated)"], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('First Dose and Second Dose'):
    st.subheader('First and Second Doses Administered by State')
    fig = px.bar(data3, x="State", y=['First Dose Administered','Second Dose Administered'], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    



if st.sidebar.checkbox('Map'):
    st.subheader('Covid19 Vaccine Map')
    fig = px.choropleth_mapbox(
     data3,
    locations="id",
    geojson = states,
    color="Total Individuals Vaccinated",
    hover_name="State",
    hover_data =['Total Individuals Vaccinated', 'Total Sessions Conducted',
       'Total Sites', 'First Dose Administered', 'Second Dose Administered',
       'Male(Individuals Vaccinated)', 'Female(Individuals Vaccinated)',
       'Transgender(Individuals Vaccinated)', 'Total Covaxin Administered',
       'Total CoviShield Administered'],   
    title="India Covid-19 Vaccine Map",
    mapbox_style="carto-positron",
    center={"lat": 24, "lon": 78},
    zoom=3,
    opacity=0.5,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)    


# # Model ml

# In[19]:


import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('covid_model7_linear_model.pkl','rb'))

st.sidebar.subheader('Model')
if st.sidebar.checkbox("Predict"):



    def predict(sessions,sites):
        inputs=np.array([[sessions,sites]]).astype(np.float64)
        prediction=model.predict(inputs)
        print(prediction)
        return float(prediction)

    def main():
        st.title("Covid19 ml model")
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">covid19 vaccination ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        sessions = st.text_input("Total no.of covid vaccination sessions to be conducted","Type Here")
        sites = st.text_input("Total no.of vaccination sites (in area,region,state)","Type Here")
    

        if st.button("Predict"):
            output=predict(sessions,sites)
            st.success('The predicted no.of vaccinated people {}'.format(output))
        

        fig = px.choropleth_mapbox(
                 data3,
                 locations="id",
                 geojson = states,
                 color="Total Individuals Vaccinated",
                 hover_name="State",
                 hover_data =['Total Individuals Vaccinated', 'Total Sessions Conducted',
                             'Total Sites', 'First Dose Administered', 'Second Dose Administered',
                             'Male(Individuals Vaccinated)', 'Female(Individuals Vaccinated)',
                             'Transgender(Individuals Vaccinated)', 'Total Covaxin Administered',
                             'Total CoviShield Administered'],   
                              title="India Covid-19 Vaccine Map",
                 mapbox_style="carto-positron",
                 center={"lat": 24, "lon": 78},
                 zoom=3,
                 opacity=0.5,
                                  )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig)    
                    
    if __name__=='__main__':
            main()


# In[ ]:





# In[ ]:





# In[ ]:




