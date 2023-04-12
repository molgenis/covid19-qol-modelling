#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import collections
from collections import Counter
import math

import sys
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config



def concat_questionnaires_filter(path_directory, directory):
    """
    Merge questionnaires.
    Get only these columns: 'project_pseudo_id', 'responsedate', 'qualityoflife'
    """
    # Empty dataframe with three columns
    all_quest = pd.DataFrame(columns=['project_pseudo_id', 'responsedate', 'qualityoflife'])
    # Concat dataframes to one dataframe (all_quest)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        # Only grab the files that start with 'covq_q_t' and end with '.csv'
        if filename.startswith("covq_q_t") and filename.endswith(".csv"):
            # Get the number of the questionnaire (this is still a string)
            number_quest = filename.split('_')[2].replace('t', '')
            # No quality of life question in questionnaire 01
            if number_quest != '01':
                # Read questionnaire number_quest
                df = pd.read_csv(f'{path_directory}{filename}')
                # Replace none_values with np.nan in df
                none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
                df[df.isin(none_value)] = np.nan
                # The strings that make up an id partly for quality of life question and date of filling in the
                # questionnaire
                string_QOL = '_qualityoflife'
                string_respons = '_responsedate_adu_q'
                # Change the IDs for the QoL question and the date the questionnaire was completed to 'qualityoflife'
                # and 'responsedate'.
                # This is so that they all have the same name.
                df.columns = ['qualityoflife' if string_QOL in x else x for x in df.columns]
                df.columns = ['responsedate' if string_respons in x else x for x in df.columns]
                # Select only these columns out of the dataframe: 'project_pseudo_id', 'responsedate', 'qualityoflife'
                df_select = df[['project_pseudo_id', 'responsedate', 'qualityoflife']]
                # Concat the questionnaire dataframe to all_quest
                all_quest = pd.concat([all_quest, df_select], ignore_index=True)  # , axis=1
    # Drop NA values  
    all_quest = all_quest.dropna()
    # Change types
    all_quest['responsedate'] = pd.to_datetime(all_quest['responsedate'])
    all_quest = all_quest.astype({'qualityoflife': 'int64'})
    return all_quest


def calculate_mean_QOL(all_quest, path_save):
    """
    Calculate the mean per date. (=df_q)
    Calculate the mean per date per person (=df_id)
    """
    # # Check duplicates
    # respons_id = all_quest.groupby(['responsedate', 'project_pseudo_id']).size().reset_index() # or count() (size())
    # check_duplicates = respons_id[respons_id['qualityoflife']> 1]

    # Calculate mean quality of life per date per date per person
    df_id = all_quest.groupby(['responsedate', 'project_pseudo_id']).mean().reset_index()
    # Save file
    df_id.to_csv(f'{path_save}df_id.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    # Calculate mean quality of life per date
    df_q = all_quest.groupby('responsedate')['qualityoflife'].mean().reset_index()
    df_q.to_csv(f'{path_save}df_q.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    # how_many = all_quest.groupby('responsedate').count()
    return df_id, df_q



def main():
    # Call get_config
    config = get_config()
    # Path to the folder containing the results of each questionnaire.
    path_directory = config['path_questionnaire_results']
    path_save = config['make_df_id']
    directory = os.fsencode(path_directory)
    # Call different functions
    all_quest = concat_questionnaires_filter(path_directory, directory)
    df_id, df_q = calculate_mean_QOL(all_quest, path_save)
    print('DONE')


if __name__ == '__main__':
    main()

