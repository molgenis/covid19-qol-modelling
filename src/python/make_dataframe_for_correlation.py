#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings
warnings.filterwarnings('ignore')



def add_weather_QOL(path_read_QOL, create_model):
    """
    Add the weather data #https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    data comes from
    Return
    final_dataframe: 
    total_df: 
    df_participants:
    """
    # Read file
    knmi_data = pd.read_excel(
        f'{path_read_QOL}Weather_data.xlsx')
    knmi_data.drop('Unnamed: 0', axis=1, inplace=True)
    knmi_data = knmi_data.replace(r'^\s*$', np.NaN, regex=True)
    knmi_data['date'] = pd.to_datetime(knmi_data['date'])
    # How the students did earlier with the average of the temperature
    avg = (knmi_data['Minimum Temperature (\u00B0C)'] + knmi_data['Maximum Temperature (\u00B0C)']) / 2
    knmi_data.insert(3, "avg_temp", avg)
    # adding .. day rolling mean weather avg
    for days in range(1,22):
        knmi_data[f'{days}day_rolling_avg_temp'] = knmi_data.rolling(days, min_periods=1)['avg_temp'].mean()
        knmi_data[f'{days}_max_temp'] = knmi_data.rolling(days, min_periods=1)['Maximum Temperature (\u00B0C)'].mean()
    # read_file 
    df = pd.read_csv(f'{create_model}num_quest_1_filter.tsv.gz', sep='\t',
                     encoding='utf-8', compression='gzip') # num_quest_1_filter, QOL_data_VL29
    # Select columns
    df = df[['project_pseudo_id', 'responsedate', 'qualityoflife']]
    # Groupby responsedate
    df['size_responsedate'] = df.groupby(['responsedate'])[["responsedate"]].transform('size')
    # df = df[(df['size_participants'] >= 15) & (df['size_responsedate'] >= 50)]
    df = df[df['size_responsedate'] >= 50]
    df['responsedate'] = pd.to_datetime(df['responsedate'])
    df.rename({'responsedate': 'date'}, axis=1, inplace=True)
    # Merge df and knmi_data
    df_participants = pd.merge(df, knmi_data, how='left', on=['date'])
    # Groupby qualityoflife and size_responsedate
    df_new = df.groupby(['date'])[["qualityoflife", 'size_responsedate']].mean().reset_index()
    df_new.columns = ['date', 'qualityoflife', 'size_responsedate']
    # Merge files
    final_dataframe = pd.merge(df_new, knmi_data, how="left", on=["date"])
    total_df = pd.merge(df_new, knmi_data, how="outer", on=["date"])
    # Drop columns
    final_dataframe = final_dataframe.drop(['station'], axis=1)
    # Split date
    final_dataframe['date'] = pd.to_datetime(final_dataframe['date'])
    final_dataframe['Year'] = final_dataframe['date'].dt.year
    final_dataframe['Month'] = final_dataframe['date'].dt.month
    final_dataframe['Day'] = final_dataframe['date'].dt.day

    return final_dataframe, total_df, df_participants


def add_hospitalization(final_dataframe, total_df, df_participants, path_read_QOL):
    """
    Add hospitalization data RIVM
    data comes from # https://data.rivm.nl/meta/srv/dut/catalog.search#/metadata/4f4ad069-8f24-4fe8-b2a7-533ef27a899f
    Return
    final_dataframe: 
    total_df: 
    df_participants:
    """
    # Read files
    hospitalization_1 = pd.read_csv(f'{path_read_QOL}COVID-19_ziekenhuisopnames.csv',sep=';', encoding='utf-8')
    hospitalization_2 = pd.read_csv(f'{path_read_QOL}COVID-19_ziekenhuisopnames_tm_03102021.csv',sep=';', encoding='utf-8')
    # Concat files
    hospitalization = pd.concat([hospitalization_1, hospitalization_2])
    hospitalization['Date_of_statistics'] = pd.to_datetime(hospitalization['Date_of_statistics'])
    hospitalization_grouped = hospitalization.groupby(['Date_of_statistics']).sum().reset_index()
    # Rename columns
    hospitalization_grouped.rename(columns={'Date_of_statistics': 'date', 'Hospital_admission_notification': 'daily_hospitalization'}, inplace=True)
    hospitalization_grouped = hospitalization_grouped[['date', 'daily_hospitalization']]
    # Merge files
    final_dataframe = pd.merge(final_dataframe, hospitalization_grouped, how="left", on=["date"])
    df_participants = pd.merge(df_participants, hospitalization_grouped, how="left", on=["date"])
    total_df = pd.merge(total_df, hospitalization_grouped, how="outer", on=["date"])
    return final_dataframe, total_df, df_participants


def add_stringency_index(final_dataframe, total_df, df_participants, path_read_QOL):
    """
    Add stringency index and other information of outwoldindata
    data comes from #https://ourworldindata.org/covid-stringency-index#learn-more-about-the-data-source-the-oxford-coronavirus-government-response-tracker
    Return
    final_dataframe: 
    total_df: 
    df_participants:
    """
    stringency_index = pd.read_csv(
        f'{path_read_QOL}owid-covid-data.csv', sep=';',
        encoding='utf-8')
    # Select on the Netherlands
    stringency_index = stringency_index[stringency_index['location'] == 'Netherlands']
    stringency_index['date'] = pd.to_datetime(stringency_index['date'], dayfirst=True)
    stringency_index.rename(columns={'hosp_patients': 'daily_hospitalization_2'}, inplace=True)
    # Merge files
    final_dataframe = pd.merge(final_dataframe, stringency_index, how="left", on=["date"])
    df_participants = pd.merge(df_participants, stringency_index, how="left", on=["date"])
    total_df = pd.merge(total_df, stringency_index, how="outer", on=["date"])
    return final_dataframe, total_df, df_participants

def sunrise_sunset(final_dataframe, total_df, df_participants, path_read_QOL):
    """ 
    Add sunrise - sunset data
    data comes from #https://www.msimons.nl/tools/daglicht-tabel/index.php?year=2020&location=Groningen
    Return
    final_dataframe: 
    total_df: 
    df_participants:
    """
    # Read data
    daylight_hours = pd.read_csv(
        f'{path_read_QOL}sunrise_sunset.csv', sep=';',
        encoding='utf-8')
    daylight_hours['date'] = pd.to_datetime(daylight_hours['date'], dayfirst=True)
    daylight_hours.dropna(inplace=True)
    daylight_hours['sunrise'] = daylight_hours['sunrise'].apply(lambda x: pd.to_datetime(x).strftime('%H:%M:%S'))
    daylight_hours['sunset'] = daylight_hours['sunset'].apply(lambda x: pd.to_datetime(x).strftime('%H:%M:%S'))
    daylight_hours['sunrise'] = pd.to_timedelta(daylight_hours['sunrise'].astype(str)).dt.total_seconds() / 3600
    daylight_hours['sunset'] = pd.to_timedelta(daylight_hours['sunset'].astype(str)).dt.total_seconds() / 3600
    # daylight_hours = sunset - sunrise
    daylight_hours['daylight_hours'] = (daylight_hours['sunset'] - daylight_hours['sunrise'])

    # .. day shift daylight hours
    for i in range(1,20,1):
        daylight_hours[f'{i}day_daylight_hours'] = daylight_hours['daylight_hours'].shift(i)
        # daylight_hours[f'{i}day_daylight_hours'] = daylight_hours.rolling(i, min_periods=1)['daylight_hours'].mean()
        
    daylight_hours=daylight_hours.drop(['sunset','sunrise'],axis=1) 
    # Merge files
    df_participants = pd.merge(df_participants, daylight_hours, how="left", on=["date"])
    final_dataframe = pd.merge(final_dataframe, daylight_hours, how="left", on=["date"])
    total_df = pd.merge(total_df, daylight_hours, how="outer", on=["date"])
    return final_dataframe, total_df, df_participants

def add_other_cat(final_dataframe, path_read_QOL):
    """
    Add financial data
    Return
    final_dataframe:
    """
    news_sentiment = pd.read_excel(
        f'{path_read_QOL}news_sentiment_data.xlsx', 'Data')
    df = pd.merge(df, news_sentiment, how='left', on=['date'])
    finacial_data = pd.read_excel(
        f'{path_read_QOL}finacial_data.xlsx')
    final_dataframe = pd.merge(final_dataframe, finacial_data, how='left', on=['date'])

    
    
    return final_dataframe

def main():
    config = get_config()
    path_read_QOL = config['data_QOL']
    # Add different data to one dataframe
    final_dataframe, total_df, df_participants = add_weather_QOL(path_read_QOL) 
    final_dataframe, total_df, df_participants = add_hospitalization(final_dataframe, total_df, df_participants, path_read_QOL)
    final_dataframe, total_df, df_participants = add_stringency_index(final_dataframe, total_df, df_participants, path_read_QOL)
    final_dataframe, total_df, df_participants = sunrise_sunset(final_dataframe, total_df, df_participants, path_read_QOL)
    df_corr = add_other_cat(final_dataframe, path_read_QOL)
    # Save file
    df_corr.to_csv(f'{path_read_QOL}df_corr.tsv.gz', 
                        sep='\t', encoding='utf-8', compression='gzip')

    print('DONE')


if __name__ == '__main__':
    main()