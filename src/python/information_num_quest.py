#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import warnings
from collections import Counter
warnings.filterwarnings('ignore')


def get_data(path_directory, information_num_quest_path):
    """
    
    """
    # Get data out of lifelines questionnairs
    df_quest = pd.DataFrame(columns=['project_pseudo_id', 'responsedate', 'gender', 'age', 'num_quest'])
    for files in os.listdir(path_directory):
        if files.startswith('cov'):
            filenum = files.split('_')[2]
            filenum = str(filenum.replace('t', ''))
            print(filenum)
            df = pd.read_csv(f'{path_directory}{files}', sep=',', encoding='utf-8')
            df = df[['project_pseudo_id', f'covt{filenum}_responsedate_adu_q_1', 'gender', 'age']]
            df.rename(columns={f'covt{filenum}_responsedate_adu_q_1': 'responsedate'}, inplace=True)
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            # Add a column with which questionnaire this answer comes from
            df['num_quest'] = filenum
            df_quest = pd.merge(df_quest, df, how='outer', on=['project_pseudo_id', 'responsedate', 'gender', 'age', 'num_quest'])
    # Save file
    df_quest.to_csv(f'{information_num_quest_path}num_quest.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    return df_quest

def filter_data(df_quest, information_num_quest_path):
    """
    
    """
    # Read file
    if df_quest.empty:
        df_quest = pd.read_csv(f'{information_num_quest_path}num_quest.tsv.gz', sep='\t',
                        encoding='utf-8', compression='gzip')
    # Groupby
    df_part = df_quest.groupby('project_pseudo_id').size().reset_index()
    df_part.rename(columns={df_part.columns[1]: "num_entered" }, inplace = True)
    df_response = df_quest.groupby('num_quest').size().reset_index()
    df_response.rename(columns={df_response.columns[1]: "num_participants" }, inplace = True)
    # Merge files
    df_quest = pd.merge(df_quest, df_part, how='outer', on=['project_pseudo_id'])
    df_quest = pd.merge(df_quest, df_response, how='outer', on=['num_quest'])
    # Filter data on num_entered more or equeal to 15
    df_filter = df_quest[df_quest['num_entered'] >= 15]
    df_response_filter = df_filter.groupby('num_quest').size().reset_index()
    df_response_filter.rename(columns={df_response_filter.columns[1]: "num_participants_filter"}, inplace = True)
    # Merge files
    df_quest = pd.merge(df_quest, df_response_filter, how='outer', on=['num_quest'])
    # Save files
    df_quest.to_csv(f'{information_num_quest_path}num_quest_filter.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    # df_quest.to_csv(f'{information_num_quest_path}num_quest_filter.tsv', sep='\t', encoding='utf-8',
    #                     index=False)
    # Select coluns
    df_quest_only = df_quest[['num_quest', 'num_participants', 'num_participants_filter']]
    df_quest_only_gender = df_quest.groupby(['num_quest', 'gender']).size().reset_index()
    df_quest_only_gender.rename(columns={df_quest_only_gender.columns[2]: "gender_count"}, inplace = True)
    df_quest_only = pd.merge(df_quest_only, df_quest_only_gender, how='outer', on=['num_quest'])
    # Filter 
    df_quest_only_filter = df_quest[df_quest['num_entered'] >= 15]
    df_quest_only_gender_filter = df_quest_only_filter.groupby(['num_quest', 'gender']).size().reset_index()
    df_quest_only_gender_filter.rename(columns={df_quest_only_gender_filter.columns[2]: "gender_count_filter"}, inplace = True)
    df_quest_only = pd.merge(df_quest_only, df_quest_only_gender_filter, how='outer', on=['num_quest', 'gender'])

    df_age = df_quest.groupby(['num_quest', 'gender']).mean().reset_index()
    df_age.rename(columns={'age': 'avg_age'}, inplace=True)
    df_quest_only = pd.merge(df_quest_only, df_age[['num_quest', 'gender', 'avg_age']], how='outer', on=['num_quest', 'gender'])
    df_quest_only.drop_duplicates(inplace=True)
    df_quest_only['num_quest'] = df_quest_only['num_quest'].astype(str)
    df_quest_only.sort_values(by='num_quest', inplace=True)

    df_quest_only.to_csv(f'{information_num_quest_path}num_quest_filter_only.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    # df_quest_only.to_csv(f'{information_num_quest_path}num_quest_filter_only.tsv', sep='\t', encoding='utf-8',
    #                     index=False)
    return df_quest_only
    
def open_file(information_num_quest_path):
    """
    
    """
    # Read file
    if df.empty:
        df = pd.read_csv(f'{information_num_quest_path}num_quest_filter_only.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    # Select
    df = df[['num_quest', 'num_participants', 'num_participants_filter']].drop_duplicates().reset_index()
    # Save
    df.to_csv(f'{information_num_quest_path}num_quest_filter_only_nogenderage.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    # df.to_csv(f'{information_num_quest_path}num_quest_filter_only_nogenderage.tsv', sep='\t', encoding='utf-8',
    #                     index=False)




def main():
    config = get_config()
    path_directory = config['path_questionnaire_results']
    information_num_quest_path = config['information_num_quest']
    df_quest = pd.DataFrame()
    df_quest_only = pd.DataFrame()
    df_quest = get_data(path_directory, information_num_quest_path)
    df_quest_only = filter_data(df_quest, information_num_quest_path)
    open_file(information_num_quest_path, df_quest_only)
    print('DONE')


if __name__ == '__main__':
    main()

