import seaborn as sns
import pandas as pd
import streamlit as st
import numpy as np
import folium
import geopandas
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from matplotlib import pyplot as plt
from numerize.numerize import numerize
  
pd.set_option('display.float_format', lambda x: '%.2f' % x)  

st.set_page_config(page_title="House Rocket Insights", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")


st.title('House Rocket Company')
st.header('Analysis Purpose')
st.markdown('This report was created for the purpose analyzing and identifying better business opportunities across all available properties in King County, Seattle.')

#==================================================================================

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data

@st.cache(allow_output_mutation=True)
def get_geofile( url ):
    geofile = geopandas.read_file(url)
 
    return geofile

def transform_data(data):
    
    #drop duplicate
    data = data.drop_duplicates(subset='id', keep='last')
    
    #remove outlier
    data = data[data['bedrooms'] != 33]
    
    #drop columns
    data.drop(columns=['sqft_living15','sqft_lot15'], inplace=True)

    return data

def set_feature(data):
    
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d')
    data['sqft_living'] = data['sqft_living'].apply(lambda x: x * 0.09290304)
    data['sqft_lot'] = data['sqft_lot'].apply(lambda x: x * 0.09290304)
    data['sqft_above'] = data['sqft_above'].apply(lambda x: x * 0.09290304)
    data['sqft_basement'] = data['sqft_basement'].apply(lambda x: x * 0.09290304)
    data['price_m2'] = data['price'] / data['sqft_lot']
    data['month'] = pd.DatetimeIndex(data['date']).month
    data['season'] = data['month'].apply(lambda x: 'autumn' if (x >=9 ) & (x <= 11) else
                                                    'spring' if (x >=3 ) & (x <= 5) else
                                                    'summer' if (x >=6 ) & (x <= 8) else
                                                    'winter' )
    
    data.rename(columns = {'sqft_living': 'm2_living','sqft_lot':'m2_lot','sqft_above':'m2_above','sqft_basement':'m2_basement'}, inplace=True) 

    return data

def overview_data(data):
    if st.checkbox('Full Dataset'):
        st.subheader("Data Dimensions")
        st.write("Number of Registers:", data.shape[0])
        st.write("Number of Attributes:", data.shape[1])
        st.dataframe(data)

    c1, c2 = st.columns((1,1)) 

    #Average Metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['m2_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    #Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    #Statistic Analysis
    num_atributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_atributes.apply(np.mean))
    mediana = pd.DataFrame(num_atributes.apply(np.median))
    std = pd.DataFrame(num_atributes.apply(np.std))
    max_ = pd.DataFrame(num_atributes.apply(np.max))
    min_ = pd.DataFrame(num_atributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['atributes', 'max', 'min', 'media', 'median', 'std']

    #plot
    c1.header('Zipcode Average')
    c1.dataframe(df, height=500)

    c2.header('Statistic Analysis')
    c2.dataframe(df1, height=500)
    
    return None

def portifolio_distribution (data, geofile):
    st.header('Selected Properties')
    

    df = data[['zipcode', 'price', 'season']].groupby(['zipcode']).median().reset_index().rename(columns ={'price': 'median_price'})
    data= pd.merge( df, data, on='zipcode', how='inner')
    data['status'] = data[['price', 'median_price', 'condition']].apply(lambda x: 1 if ((x['price'] <= x['median_price']) & (x['condition'] > 3))
                                                                                    else 0 if (x['price'] > x['median_price']) 
                                                                                    else 0 , axis=1)

    df2 = data[(data['status'] == 1 )] [['id', 'date', 'zipcode', 'price' , 'median_price','status', 'condition','season', 'lat', 'long',]].reset_index(drop=True)
    

    #Selling price suggestion

    df3 = df2[['id','date','zipcode','price','season', 'lat', 'long']].copy()
    df_sell= df3[['price', 'zipcode','season']].groupby(['zipcode', 'season']).median().reset_index().rename(columns ={'price': 'median_price'})
    df_sell = pd.merge(df3, df_sell, on=['zipcode', 'season'], how='inner')

    df_sell['sugg_price'] = df_sell[['price', 'median_price']].apply(lambda x: (x['price']*1.3) if x['price'] < x['median_price']
                                                                                        else (x['price']*1.1), axis=1) 

    df_sell['profit'] = df_sell['sugg_price'] - df_sell['price']
    
    df_sell = df_sell[['id', 'date', 'zipcode','price','median_price', 'sugg_price','profit', 'season', 'lat', 'long']]

    c1, c2= st.columns((1,1))

    with c2:

        
        f_buying_price = st.slider('Select maximum price',
                                            int(df_sell['price'].min()),
                                            int(df_sell['price'].max()), key='price bus',
                                            value=int(df_sell['price'].max()))
        f_expected_profit = st.slider('Select maximum expected profit',
                                                int(df_sell['profit'].min()),
                                                int(df_sell['profit'].max()),
                                                value=int(df_sell['profit'].max()))
            
        f_b = df_sell[(df_sell['price'] <= f_buying_price) & (df_sell['profit'] <= f_expected_profit)]

    with c1:
        
        f_zipcode = st.multiselect('Type or select zipcodes', f_b['zipcode'].sort_values(ascending=True).unique())
        f_id = st.multiselect('Type or select properties ID', f_b['id'].sort_values(ascending=True).unique())

        
        if (f_id != []) & (f_zipcode != []):
            f_b2 = f_b.loc[(f_b['id'].isin(f_id)) & (f_b['zipcode'].isin(f_zipcode)), :]
                   
        elif (f_id != []) & (f_zipcode == []):
            f_b2 = f_b.loc[f_b['id'].isin(f_id), :]
            
        elif (f_id == []) & (f_zipcode != []):
            f_b2 = f_b.loc[f_b['zipcode'].isin(f_zipcode), :]
                
        else:
            f_b2 = f_b.copy()

    c3, c4= st.columns((1,1))

    #BASE MAP - Folium
    with c3:

        c3.subheader('Properties distribution')
        df_map = folium.Map(location = [data['lat'].mean(), data['long'].mean()], default_zoom_start=15)
   
        marker_cluster = MarkerCluster().add_to(df_map)

        for i, row in f_b2.iterrows():
            folium.Marker( [row['lat'], row['long'] ],
            popup='Avaiable since{0} for US$ {1}. ID: {2} ZIPCODE {3} PROFIT {4} '.format( row['date'],
                                                                                            row['price'],
                                                                                            row['id'],
                                                                                            row['zipcode'],
                                                                                            row['profit'])).add_to(marker_cluster)
                                                                                                            
                        
        geofile = geofile[geofile['ZIP'].isin( f_b2['zipcode'].tolist() )]
        
        df_map.choropleth( data = df_sell,
                                geo_data = geofile,
                                columns=['zipcode', 'profit'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity = 0.7,
                                line_opacity = 0.2,
                                legend_name='Expected Profit')

        folium_static(df_map)

    with c4:           

        
        c4.subheader('Properties Information')
        c4.dataframe(f_b2, height=450)

    st.subheader('Expected profit by season')
    df4 = df_sell[['profit', 'season']].groupby(['season']).sum().reset_index()
    fig = plt.figure(figsize=(10,5))
    sns.barplot(data=df4, x='season', y='profit')
    st.pyplot(fig, use_container_width= True)   

    
    #sidebar general information
    investment = df_sell['price'].sum()
    returned = df_sell['sugg_price'].sum()
    profit = df_sell['profit'].sum()
    perc = (profit/ investment) * 100
    
    st.sidebar.image('houserocket.png')
    st.sidebar.title('General Information')
    st.sidebar.metric(label='Maximum Expected Profit', value=numerize(profit), delta=numerize(perc) + "%")
    st.sidebar.metric(label='Maximum Value Invested', value=numerize(investment))
    st.sidebar.metric(label='Maximum Value Returned', value=numerize(returned))
   


    return None

def hipoteses (data):

    st.header('Hypothesis Analysis')
    c1, c2 = st.columns((1,1))
    
    c1.markdown('##### H1: Properties with waterview are on average 50% more expensive.')
    h1 = data[['price', 'waterfront']].groupby('waterfront').mean().reset_index()
    perc = (h1.loc[1, 'price'] - h1.loc[0, 'price']) / h1.loc[0, 'price']
    fig1 = plt.figure(figsize=(10,5))
    sns.barplot(data=h1, x='waterfront', y='price')
    c1.pyplot(fig1, use_container_width= True)
    c1.write(f'H1: True. Properties with water view are on average {perc:.2%} more expensive.')

    df_aux = data.copy()
    c2.markdown('##### H2: Properties with basement are on average 20% more expensive.')
    df_aux['basement'] = df_aux['m2_basement'].apply(lambda x: 0 if x == 0 else 1)
    h2 = df_aux[['price', 'basement']].groupby('basement').mean().reset_index()
    perc = (h2.loc[1, 'price'] - h2.loc[0, 'price']) / h2.loc[0, 'price']
    fig2 = plt.figure(figsize=(10,5))
    sns.barplot(data=h2, x='basement', y='price')
    c2.pyplot(fig2, use_container_width= True)
    c2.write(f'H2: True. Properties with basement are on average {perc:.2%} more expensive.')

    st.markdown('---')

    c3, c4 = st.columns((1,1))

    c3.markdown('##### H3: Properties built with basements decreased over the years.')
    b1 = data[data['m2_basement']!=0][['m2_basement','yr_built']].groupby('yr_built').count().reset_index()
    b2 = data[data['m2_basement']==0][['m2_basement','yr_built']].groupby('yr_built').count().reset_index()
    h3 = pd.merge(b1, b2, on='yr_built', how='left').rename(columns={'m2_basement_x': 'with_basemement', 'm2_basement_y': 'without_basement'})
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(h3['yr_built'], h3['with_basemement'], color='blue', label='With basement')
    plt.plot(h3['yr_built'], h3['without_basement'], color='red', label='Without basement')
    plt.legend()
    c3.pyplot(fig3, use_container_width= True)
    c3.write('H3: True. Properties built with basements decreased after 1980.')

    c4.markdown('##### H4: Properties have bigger interior living space along construction year timeline.')
    h4 = data[['m2_living', 'yr_built']].groupby('yr_built').mean().reset_index()
    fig4 = plt.figure(figsize=(10,5))
    sns.lineplot(data=h4, x='yr_built', y='m2_living')
    c4.pyplot(fig4, use_container_width= True)
    c4.write('H4: Properties have bigger interior living space incresead after 1940.')

    st.markdown('---')

    c5, c6 = st.columns((1,1))

    c5.markdown('##### H5: Properties available during summer are more expensive.')
    h5 = data[['season', 'price']].groupby('season').mean().reset_index()
    fig5 = plt.figure(figsize=(10,5))
    sns.barplot(data=h5, x='season', y='price')
    c5.pyplot(fig5, use_container_width= True)
    c5.write('H5: False. Properties available during spring are more expensive on average than other seasons.')

    c6.markdown('##### H6: Properties that have not been renovated are 30 percent more cheaper.')
    df_aux2 = data[['price', 'yr_renovated']].copy()
    df_aux2['renovated'] = df_aux2.apply(lambda x: 'No' if x['yr_renovated']==0 else 'Yes', axis=1)
    h6 = df_aux2[['price', 'renovated']].groupby('renovated').mean().reset_index()
    perc = (h6.loc[1, 'price'] - h6.loc[0, 'price']) / h6.loc[0, 'price']
    fig6 = plt.figure(figsize=(10,5))
    sns.barplot(data=h6, x='renovated', y='price')
    c6.pyplot(fig6, use_container_width= True)
    c6.write(f'H6: True. Properties that have not been renovated are {perc:.2%} more cheaper.')
    
###################################################

if __name__ == '__main__':

    #data extration
    path = 'kc_house_data.csv'
    data = get_data(path)
    url = 'King_county.geojson'
    geofile = get_geofile(url)

    #transformation
    data = transform_data(data)
    data = set_feature(data)
    overview_data(data)
    portifolio_distribution(data, geofile)
    hipoteses(data)

###################################################

st.markdown('---')
st.header('Additional Information')
st.markdown("This report was created by a request from House Rocket's CEO to visualize all properties available to be bought at King County, Seatle.")
st.markdown("This data visualization is part of House Rocket Insights Project made by Daniel Salvatori.")
st.markdown("For more information about business context you can check the code for this streamlit on [github](https://github.com/Daniel-Salvatori/P01_house_rocket_insights).")
st.markdown('---')
st.header("Contacts")
st.markdown('''[🤝 Linkedin](https://www.linkedin.com/in/daniel-salvatori/) │ 
               [📁 GitHub](https://github.com/daniel-salvatori)''' )
st.markdown('---')

