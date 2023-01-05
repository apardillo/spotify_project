#%%
import numpy as np 
import pandas as pd 
#import seaborn as sns
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _1_load_data import Load_Data
from _2_chart_functions import Chart_Functions
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing

import warnings
import pickle
warnings.filterwarnings('ignore')


class Graphs:

    def __init__(self):

        dl=Load_Data()
        self.charts_df=dl.charts_df
        self.tracks_df=dl.tracks_df
        self.streams_df=dl.streams_df
        self.zoo_tracks=dl.zoo_tracks
        self.artists_df=dl.artists_df

        cf=Chart_Functions()
        self.cf=cf

        #Config
        self.config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            'height': 450,
            'width': 500,
            'scale': 2 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

        #Color Compilation

        self.red="#DD634F"
        self.maroon="#AB5474" 
        self.pink="#ffcccc"

        self.green="#4fdd63"
        #self.green="#28d140"
        self.navy_green="#448299"

        self.light_blue="#75B9DF"
        self.blue="#3558C9"
        self.navy_blue="#2D3B6A"
        self.purple="#646EC8"

        self.grey="#E9E3E2"


        #Custom Color pallettes
        self.custom_colors_yellow=["#f9f8ef","#ddd6a4","#c8bd6c","#a89c3f"]
        self.custom_colors_red=["#FFF0F5", "#FFB6C1", "red", "#8B0000", "#800000"]


        #Background Color
        self.bgcolor=self.navy_blue  ##F8FBFB"  -#FAFAF9  #F0F2F6 #EDE9EA
        self.gridcolor="#D4E5E6"


        self.line_color1=self.pink
        self.line_color2=self.light_blue
        self.line_color3=self.red


        #Annotation Colors
        self.ann_text_color="#ffffff"
        self.ann_arrow_color="#636363"
        self.ann_border_color="#c7c7c7"
        self.ann_bg_color="#ff7f0e"

        self.ann_bg_color2="#63A2B6"
        self.ann_bg_color3="black"
        self.ann_bg_color4="red"



        #Style- CSS #FAFAF9

    #Annotate- Non Subplot
    def annotate(self,fig,x,y,text,axv,axy, bg_color, border_color,showarrow=True):
        border_color=border_color
        bg_color=bg_color
        
        
        fig.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=text,
        showarrow=showarrow,
        font=dict(
            family="Courier New, monospace",
            size=12, 
            color=self.ann_text_color
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=self.ann_arrow_color,
        ax=axv,
        ay=axy,
        bordercolor=border_color,
        borderwidth=2,
        borderpad=4,
        bgcolor=bg_color,
        opacity=0.8
        )


    #Annotate Subplot
    def annotate_subplot_box(self,fig,x,y,text,axv,axy,xref,yref, bgcolor=1,showarrow=True):
        if bgcolor==1:
            bgc=self.line_color3
        elif bgcolor==2:
            bgc=self.ann_bg_color4
        elif bgcolor==3:
            bgc=self.ann_bg_color3
        elif bgcolor==4:
            bgc=self.ann_bg_color4


        fig.add_annotation(
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        text=text,
        showarrow=showarrow,
        font=dict(
            family="Courier New, monospace",
            size=12, 
            color=self.ann_text_color
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=self.ann_arrow_color,
        ax=axv,
        ay=axy,
        bordercolor=self.ann_border_color,
        borderwidth=2,
        borderpad=4,
        bgcolor=bgc,
        opacity=0.8
        )

    def annotate_subplot_text(self,fig,x,y,text,axv,axy,xref,yref, bgcolor="default",showarrow=False):

        if bgcolor=="default":
            color=self.line_color1
        else:
            color=bgcolor

        fig.add_annotation(
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        text=text,
        showarrow=showarrow,
        font=dict(
            family="sans-serif",
            size=14, 
            color=color,
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        ax=axv,
        ay=axy,
        opacity=1
        )


    #Actual Plotting
    def plot_1(self,artist):
        # charts_df=self.charts_df
        # tracks_df=self.tracks_df
        #streams_df=self.streams_df

        #artist="I Belong to the Zoo"
        data1=self.streams_df[self.streams_df.artist==artist]['streams'].resample("MS").sum()/1000000
        #data2=streams_df[streams_df.artist==artist]['track_id'].resample("MS").count()
        data2=self.streams_df[self.streams_df.artist==artist]['streams'].resample("MS").sum().cumsum()/1000000
        data3=data1.pct_change()
        data4=self.streams_df[self.streams_df.artist==artist]['position'].resample("MS").mean()




        fig = make_subplots(rows=2, cols=2,
                            #vertical_spacing = 0.04,
        subplot_titles=("Monthly Streams (in Millions)","Cumulative Streams (in Millions)","Monthly % Change","Average Position in the  Top 200"))


        marker_color=self.maroon
        opacity=1

        fig.add_trace(go.Scatter(x=data1.index[:-1], y=data1.values[:-1],name="", opacity=opacity, marker_color=marker_color,showlegend=False,),row=1, col=1)
        fig.add_trace(go.Scatter(x=data2.index[:-1], y=data2.values[:-1],name="", opacity=opacity, marker_color=marker_color,showlegend=False,),row=1, col=2)
        fig.add_trace(go.Scatter(x=data3.index[:-1], y=data3.values[:-1],name="", opacity=opacity, marker_color=marker_color,showlegend=False,),row=2, col=1)
        fig.add_trace(go.Scatter(x=data4.index[:-1], y=data4.values[:-1],name="", opacity=opacity, marker_color=marker_color,showlegend=False,),row=2, col=2)


        self.cf.update_layout(fig,"",500,800,bgcolor="white",font_color="black",font_size=10,show_ygrid=False,title_font_size=15,title_y=0.95)
        
        #left,right, bottom, top
        self.cf.update_layout_margin(fig,0,50, 0, 50)

        fig.update_yaxes(showgrid=True, 
                    gridwidth=0.4, 
                    gridcolor="#d8d8d8", 
                    griddash='dot',
                    zeroline=False,
                    )

        fig.update_yaxes(autorange="reversed",row=2,col=2)
        #fig.update_xaxes(range=[datetime.datetime(ys, ms, 5), datetime.datetime(ye, me, 5)])

        st.plotly_chart(fig,use_container_width=True)


    def plot_2(self):
        zoo_tracks=self.zoo_tracks
        #Scaling
        scaler = MinMaxScaler()
        zoo_tracks['loudness'] = scaler.fit_transform(zoo_tracks[['loudness']])
        zoo_tracks['tempo'] =  scaler.fit_transform(zoo_tracks[['tempo']])


        # Convert to long format
        audio_feature_columns=['tempo','acousticness','danceability', 
                'loudness', 'energy',
            'valence', 'liveness', 
            ]

        long_df = pd.melt(
            zoo_tracks,
            id_vars=['track_name'],
            value_vars=audio_feature_columns
        )


        data1=long_df[(long_df.track_name=="Sana - Single Version") | (long_df.track_name=="Balang Araw - Single Version")]
        data2=long_df[~((long_df.track_name=="Sana - Single Version") | (long_df.track_name=="Balang Araw - Single Version"))]

        audio_feature_columns=['tempo','acousticness','danceability', 
                'loudness', 'energy',
            'valence', 'liveness', 
            ]

        fig = make_subplots(rows=1, cols=1,
                            #vertical_spacing = 0.04,
        subplot_titles=("Comparison of Audio Features- Hit Songs vs Other Songs"))

        fig.add_trace(go.Box(x=data1.variable, y=data1.value,name="Hit Songs",marker_color=self.maroon))
        fig.add_trace(go.Box(x=data2.variable, y=data2.value,name="Other Songs",marker_color=self.navy_green))
            




        fig.update_traces(
        # hovertemplate=None,
        hoverinfo='skip'
        )


        self.cf.update_layout(fig,"",450,1200,bgcolor="grey",font_color="black",font_size=15,show_ygrid=False,title_font_size=15,title_y=0.95)
        fig.update_layout(
            yaxis_title='Audio Features',
            boxmode='group', # group together boxes of the different traces for each value of x
            paper_bgcolor="white",#F8FBFB
            plot_bgcolor="#fdfcf9",
            
        )

        self.cf.update_layout_margin(fig,0,0,0,0)
        self.cf.update_layout_legend(fig,0.85,0.95)
        #fig.update_xaxes(showgrid=True,zeroline=True)
        #fig.show()
        st.plotly_chart(fig,use_container_width=True)


    def plot_3(self):
        artists_df=self.artists_df
        artists_df["OPM"]=np.where(artists_df.genres.str.contains('opm'),1,0)

        streams_df_opm=pd.merge(self.streams_df.reset_index(),artists_df[["artist_id","OPM"]], on="artist_id", how="left")
        streams_df_opm.set_index("date")
        streams_df_opm=streams_df_opm[streams_df_opm.OPM==1]
        streams_df_opm.set_index('date',inplace=True)
        streams_df_opm.head()

        artist_dict=pd.Series(self.tracks_df.artist_name.values,index=(self.tracks_df.artist_id)).to_dict()

        df_3_groups = streams_df_opm\
            .groupby(['artist_id'])['streams']\
            .resample('M').sum().reset_index()\
            .sort_values('streams', ascending=False)
        #df_ed = df_ed.set_index('date')
        df_3_groups['album_name'] = df_3_groups['artist_id'].apply(lambda x:
                                                    'Big3' if ( (artist_dict.get(x) == 'Arthur Nery') | \
                                                                (artist_dict.get(x) == 'Adie') |
                                                                (artist_dict.get(x) == 'Zack Tabudlo') ) else
                                                    'Dominant Bands' if ((artist_dict.get(x) == 'Ben&Ben') |
                                                                (artist_dict.get(x) == 'NOBITA'))
                                                                else 'OPM Others')

        #df_3_groups = df_3_groups.set_index('date')
        df_3_groups=df_3_groups.groupby(["date","album_name"])[["streams"]].sum().reset_index()
        total=df_3_groups.groupby(["date"])[["streams"]].sum()
        data1=df_3_groups[df_3_groups.album_name=="OPM Others"]
        data1=pd.merge(data1,total.reset_index(), on="date",how="left")
        #data1.drop("album_name_y", inplace=True,axis=1)
        data1.columns=["date","album_name","streams","total_streams"]
        data1["streams_normalized"]=data1.streams/data1.total_streams*100


        data2=df_3_groups[df_3_groups.album_name=="Dominant Bands"]
        data2=pd.merge(data2,total.reset_index(), on="date",how="left")
        #data1.drop("album_name_y", inplace=True,axis=1)
        data2.columns=["date","album_name","streams","total_streams"]
        data2["streams_normalized"]=data2.streams/data2.total_streams*100


        data3=df_3_groups[df_3_groups.album_name=="Big3"]
        data3=pd.merge(data3,total.reset_index(), on="date",how="left")
        #data1.drop("album_name_y", inplace=True,axis=1)
        data3.columns=["date","album_name","streams","total_streams"]
        data3["streams_normalized"]=data3.streams/data3.total_streams*100

        fig = make_subplots(rows=1, cols=1,
                            #vertical_spacing = 0.04,
        subplot_titles=("", ""))

        color1=self.maroon
        color2=self.light_blue
        color3=self.grey


        fig.add_trace(
            go.Scatter(x=data3.date, y=data3.streams_normalized.tolist(), name="Big3", 
                    line_color=color1, 
                    #legendgroup=2,
                    fill='tozeroy',fillcolor=color1,
                    stackgroup=1
                    ),
                    row=1, col=1)



        fig.add_trace(
            go.Scatter(x=data2.date, y=data2.streams_normalized.tolist(), name="Dominant Bands", 
                    line_color=color2, 
                    #legendgroup=2,
                    fill='tonexty',fillcolor=color2,
                    stackgroup=1
                    ),
                    row=1, col=1)


        fig.add_trace(
            go.Scatter(x=data1.date, y=data1.streams_normalized.tolist(), name="OPM Others", 
                    line_color=color3, 
                    #legendgroup=2,
                    fill='tonexty',fillcolor=color3,
                    stackgroup=1,
                    opacity=1,
                    ),
                    row=1, col=1)



        fig.update_yaxes(range=[0, 100], row=1, col=1)
        fig.update_xaxes(range=["2021-01-01", "2022-10-01"], row=1, col=1)


        #self.cf.update_layout(fig,"Streams- OPM Artists in the Charts (in Percent)",500,800,bgcolor="white",font_color="black",font_size=20,show_ygrid=False,title_font_size=30,title_y=0.95)
        #self.cf.update_layout_margin(fig,0,0,0,75)

        self.cf.update_layout(fig,"Stream Composition (In Percent)",500,800,bgcolor="white",font_color="black",font_size=18,show_ygrid=False,title_font_size=30,title_y=0.95)
        self.cf.update_layout_margin(fig,0,0,0,60)

        self.cf.update_layout_legend(fig,0.04,0.95)
        
        fig.update_layout(
            title_font_family="Droid Sans, sans-serif",
            title_font_color="black",

        )
        st.plotly_chart(fig,use_container_width=True)



    def plot_heatmap(self,artist,figx,figy,shrink,colors):
        streams_df_no_dup=self.streams_df.copy()
        #streams_df_no_dup["track_name"]=streams_df_no_dup["track_name"].str.strip()
        df_artist = streams_df_no_dup[streams_df_no_dup['artist'] == artist].groupby('track_name')[['streams']]\
        .resample('M').sum()
        df_artist = df_artist.reset_index()
        df_artist['track_name'] = df_artist['track_name'].apply(lambda x: x.split('(')[0])\
            .apply(lambda x: x.split(' - ')[0])
        df_artist["track_name"]=df_artist["track_name"].str.strip()
        
        df_artist=df_artist.groupby(["date","track_name"])[["streams"]].sum().reset_index()
        #print(df_artist)

        #------------------------------------------------------
        arr_df = df_artist.pivot(index='track_name', columns='date', values='streams')
        # divide by 1M to show streams in millions
        arr_df = arr_df/1000000
        arr_df.fillna(0, inplace=True)
        arr_df['total_streams'] = arr_df.sum(axis=1)
        #arr_df = arr_df.sort_values('total_streams',ascending=False)
        arr_df

        #----------------------------------------------------------------------------
        plt.figure(figsize=(figx, figy),dpi=150)
        ax = plt.subplot(111)

        # get all month columns and specify format for xticks
        moncols = arr_df.columns[:-1]
        yymm_cols = pd.Series(moncols.values).apply(lambda x: x.strftime('%Y-%m'))

        sns.heatmap(arr_df[moncols], ax=ax,
                    #vmin=0, vmax=0.5,
                    #cmap='Greens',
                    cmap=sns.color_palette(colors),
                    #cmap=ccmap,
                    cbar_kws={'label': 'million streams', 'ticks': np.arange(0, 20, 1),"shrink": shrink},
                    xticklabels=yymm_cols, yticklabels=True, linecolor='0.8',linewidths=0.1)

        plt.ylabel('')
        plt.xlabel('')
        plt.title("Heatmap-Streams per Song", fontsize=20)
        
        #plt.show()
        st.pyplot(plt)


    def plot_4(self):
        #Color Pallette
        color1="#F2EFDC"
        color2="#1F1F1F"
        color3="#FFFFFF"

        #Secondary Colors
        color4="skyblue"
        color5="#58D68D"
        color6="#F5CBA7"
        color7="#F9E79F"
        color8="#C39BD3"
        color9="FDD4FD"
        tracks_dict=pd.Series(self.tracks_df.track_name.values,index=(self.tracks_df.track_id)).to_dict()
        top5_names=["Arthur Nery", "Adie","Zack Tabudlo","Ben&Ben","NOBITA"]
        artist_name=["I Belong to the Zoo"]
        comb_names=top5_names
        mainstay_df=pd.read_csv("data/tracks_data.csv")

        scaler = MinMaxScaler()
        mainstay_df['loudness'] = scaler.fit_transform(mainstay_df[['loudness']])
        mainstay_df['tempo'] =  scaler.fit_transform(mainstay_df[['tempo']])
        features = ['danceability', 'energy','valence', 'loudness' , 'tempo', 'liveness','acousticness',]  

        columns_to_view = ['artist_name', 'track_name'] + features
        df_features = mainstay_df[columns_to_view].copy()

        #df_features['artist'] = [artist if artist in comb_names else 'all else'
                            #for artist in df_features['artist_name'].values]
            
        # set multiindex
        df_features = df_features.set_index(['track_name', 'artist_name'])

        # reshape by pd.stack to achieve shape demanded by boxplot
        df_features_stacked = pd.DataFrame({'value': df_features.stack()})
        df_features_stacked = df_features_stacked.reset_index()
        df_features_stacked = df_features_stacked.rename(columns={'level_2': 'feature'})

        df_features_stacked['artist_name']=df_features_stacked['artist_name'].apply(lambda x: x if (x=="I Belong to the Zoo") else ( \
                                                                                    "Big3" if ((x=="Arthur Nery") | \
                                                                                (x=="Adie") | (x=="Zack Tabudlo")) 
                                                                                                    else "Dominant Bands"))
        plt.figure(figsize=(15, 6), dpi=200)
        ax = plt.subplot(111)

        sns.boxplot(data=df_features_stacked, x='feature', y='value',  hue='artist_name', ax=ax,
                    hue_order=['I Belong to the Zoo',"Big3","Dominant Bands"], palette=[color4,color1,color5])

        #)#


        ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.1), frameon=False, ncol=3)

        #'''

        ax.set_ylim([-0.0999, 1.2])
        #text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))


        ax.text(0.2, 0.95,'Similar',fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes)

        ax.text(0.68, 0.95,'Different',fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes)


        #ax.axvspan(2.5, 6.7, facecolor=color6, alpha=0.15)
        ax.axvspan(2.5, 4.5, facecolor="#C39BD3", alpha=0.15)
        #ax.axvspan(4.6, 5.45, facecolor=color6, alpha=0.15)
        ax.axvspan(1.6, 2.5, facecolor="#F5CBA7", alpha=0.15)
        plt.axvline(x=2.5)
        #'''
        ax.set_title("Comparison of Audio Features- I Belong to the Zoo vs Mainstay Artists",fontsize=15)
        st.pyplot(plt)


    def format_data(self,genre_groups,genre):
        #genre_groups = pickle.load(open('data/genre_groups.pkl', "rb"))
        total=genre_groups.groupby(["date"])[["OPM"]].sum()
        data1=genre_groups[genre_groups.predicted_genre==genre]
        data1=pd.merge(data1,total.reset_index(), on="date",how="left")
        #data1.drop("album_name_y", inplace=True,axis=1)
        data1.columns=["date","predicted_genre","genre_count","total_streams"]
        data1["streams_normalized"]=data1.genre_count/data1.total_streams*100
        return data1


    def add_trace(self,fig,data,genre_title,color,fill):
        data_=data.set_index("date")[['genre_count']].resample("MS").mean()
        #data_=data
        fig.add_trace(
        go.Scatter(x=data_.index[:-1], y=data_.genre_count.tolist()[:-1], name=genre_title, 
                line_color=color, 
                #legendgroup=2,
                #fill=fill,fillcolor=color,
                #stackgroup=1
                ),
                row=1, col=1)

    def genre_plot1(self):
        genre_groups = pickle.load(open('data/genre_groups.pkl', "rb"))
        data1=self.format_data(genre_groups,"opm acoustic")
        data2=self.format_data(genre_groups,"opm jazz")
        data3=self.format_data(genre_groups,"opm rap")
        data4=self.format_data(genre_groups,"opm rock")
        data5=self.format_data(genre_groups,"opm dance")
        data6=self.format_data(genre_groups,"opm reggae")

        fig = make_subplots(rows=1, cols=1,
                            #vertical_spacing = 0.04,
        subplot_titles=("", ""))

        color1=self.maroon
        color2=self.light_blue
        color3=self.grey
        color4=self.navy_green
        color5="green"
        color6="black"






        self.add_trace(fig,data1,'OPM Acoustic',color1,'tozeroy')
        self.add_trace(fig,data4,'OPM Rock',color4,'tonexty')
        self.add_trace(fig,data3,'OPM Rap',"purple",'tonexty')

        self.add_trace(fig,data2,'OPM Jazz',color2,'tonexty')

        self.add_trace(fig,data5,'OPM Dance',color5,'tonexty')
        self.add_trace(fig,data6,'OPM Reggae',color6,'tonexty')
                    


        #fig.update_yaxes(range=[0, 100], row=1, col=1)
        #fig.update_xaxes(range=["2021-01-01", "2022-10-01"], row=1, col=1)
        self.cf.update_layout(fig,"Average Daily Chart Appearances by Genre",700,850,bgcolor="white",font_color="black",font_size=15,show_ygrid=True,title_font_size=30,title_y=0.95)
        self.cf.update_layout_margin(fig,0,0,0,75)
        #cf.update_layout_legend(fig,0.04,0.95)
        fig.update_yaxes(showgrid=True, 
                    gridwidth=0.5, 
                    gridcolor=self.grey, 
                    griddash='dash',
                    zeroline=False)
        st.plotly_chart(fig,use_container_width=True)

    def forecasting(self,spec_genre):
        spec_dpi=150
        genre_groups = pickle.load(open('data/genre_groups.pkl', "rb"))
        genre_groups.groupby(['predicted_genre'])[["OPM"]].sum().sort_values(by='OPM',ascending=False)
        #spec_genre="opm rock"

        data=genre_groups[genre_groups.predicted_genre==spec_genre]
        #data.date.values.min(),data.date.values.max()
        data.set_index("date",inplace=True)
        data.columns=["predicted_genre","streams"]
        
        train_start_date_range='2017-01-01'
        train_end_date_range='2020-08-08'
        test_start_data_range='2020-08-08'
        test_end_data_range='2022-09-15'
        
        forward_test_end_data_range='2024-09-15'
        
        train_df = data[train_start_date_range:train_end_date_range]
        test_df = data[test_start_data_range:test_end_data_range]
        
        #data_=data.copy()
        last_date = data.index[-1]
        last_streams = data.streams[-1]

        # Generate a sequence of dates for 2 years starting from the last date
        new_dates = pd.date_range(start=last_date, periods=24*12, freq='MS')

        # Create a new DataFrame with the new dates and fill with zeros
        new_data = pd.DataFrame(last_streams, index=new_dates, columns=data.columns)
        
        # Concatenate the existing data and the new data
        data_ = pd.concat([data, new_data])
                                
        forward_test_df = data_[test_start_data_range:forward_test_end_data_range]
        #display(forward_test_df)
        
        train_df_ = train_df.copy()
        test_df_ =  test_df.copy()
        forward_test_df_=forward_test_df.copy()
        train_df_.index = pd.DatetimeIndex(train_df.index).to_period('D')
        test_df_.index = pd.DatetimeIndex(test_df.index).to_period('D')
        forward_test_df_.index = pd.DatetimeIndex(forward_test_df.index).to_period('D')
        
        
        
        #PACF
        #print("ACF and PACF")
        fig = plt.figure(figsize=(15,5),dpi=spec_dpi)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        acf = plot_acf(train_df['streams'].interpolate().diff()[1:], lags=50, ax=ax1)
        pacf = plot_pacf(train_df['streams'].interpolate().diff()[1:], lags=50, ax=ax2)
        st.pyplot(fig)
        
        
        fig = plt.figure(figsize=(15,3),dpi=spec_dpi)
        plt.plot(train_df['streams'], color='C0', label='train')
        plt.plot(test_df['streams'], color='C7', label='test')
        plt.legend()
        plt.ylabel("Count in Charts")
        #print("Train Test Split")
        plt.title("Train Test Split")
        #plt.show()
        st.pyplot(fig)
        
        #print("")
        #print("Windowed Average Approach")
        forecast_df = data.rolling(7).mean().shift(1)[test_start_data_range:test_end_data_range]
        # plot the forecast
        fig = plt.figure(figsize=(15,2.5),dpi=spec_dpi)
        plt.plot(train_df['streams'], color='C0', label='train')
        plt.plot(test_df['streams'], color='C7', label='test')
        plt.plot(forecast_df['streams'], color='C1', label='forecast')
        plt.legend()
        plt.ylabel("Count in Charts")
        plt.title("Windowed Average Approach")
        #plt.show()
        st.pyplot(fig)
        
        print("")
        #print("Trend and Seasonality Approach")
        
        model_fit = ExponentialSmoothing(train_df_['streams'],seasonal_periods=7 ,trend='add', seasonal='add').fit()
        forecast_df = pd.DataFrame(model_fit.forecast(len(forward_test_df_)).values, index=forward_test_df_.index,\
                                columns=['streams']) 
        #display(forecast_df)


        #plot the fitted training data
        fig = plt.figure(figsize=(15,2.5),dpi=100)

        plt.plot(train_df['streams'], color='C0', label='train')
        plt.plot(test_df['streams'], color='C7', label='test')
        plt.plot(forecast_df['streams'], color='C1', label='forecast')
        plt.legend()
        plt.ylabel("Count in Charts")
        plt.title("Trend and Seasonality Approach (7)")
        #plt.show()
        st.pyplot(fig)


        print("")
        #print("ARIMA")
        model = ARIMA(train_df_[["streams"]], order=(7, 1, 7))  
        model_fit = model.fit() 
        train_fit_df = pd.DataFrame(model_fit.fittedvalues, columns=['streams'])
        forecast_df = pd.DataFrame(model_fit.forecast(len(forward_test_df_),dynamic=True).values, index=forward_test_df_.index,\
                                columns=['streams'])
        #plot the fitted training data
        fig = plt.figure(figsize=(15,2.5),dpi=spec_dpi)

        plt.plot(train_df['streams'], color='C0', label='train')
        plt.plot(test_df['streams'], color='C7', label='test')
        plt.plot(forecast_df['streams'], color='C1', label='forecast')
        plt.legend()
        plt.ylabel("Count in Charts")
        plt.title("ARIMA (7,1,7)")
        #plt.show()
        st.pyplot(fig)






#%%
if __name__=='__main__':
    
    #g=Graphs()
    pass



# %%
