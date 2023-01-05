 
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', 500)
#import matplotlib
import time
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0])
import pickle
import numpy as np
import datetime as dt

pd.options.display.float_format = '{:,.0f}'.format


class Load_Data:

    def __init__(self):
        # read and process the charts dataset
        charts_df = pd.read_csv('data/spotify_daily_charts.csv')


        # transform date column into a datetime column
        charts_df['date'] = pd.to_datetime(charts_df['date'])



        # read and process the tracks dataset
        tracks_df = pd.read_csv('data/spotify_daily_charts_tracks.csv')


        # merge the charts and tracks dataset
        streams_df = charts_df.merge(tracks_df, on='track_id', how='left')
        streams_df = streams_df.drop(columns='track_name_y')
        streams_df = streams_df.rename(columns={'track_name_x': 'track_name'})
        streams_df['date']=pd.to_datetime(streams_df['date'])
        streams_df.set_index("date", inplace=True)

        zoo_tracks=pd.read_csv('data/zoo_tracks_data.csv')

        self.charts_df=charts_df
        self.tracks_df=tracks_df
        self.streams_df=streams_df
        self.zoo_tracks=zoo_tracks
        self.artists_df = pd.read_csv('data/spotify_daily_charts_artists_edited.csv')



if __name__=='__main__':
    
    #dl=Load_Data()
    #print(dl.pp_raw_hotel_data())
    pass



