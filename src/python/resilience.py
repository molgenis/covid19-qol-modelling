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
import collections
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def variables_data(path_questionnaire_variables):
    """
    Get the resilience questions ID out of the questionnaires
    resilience_id: The ID of the resilience question
    """
    resilience_ids = set()

    for files in os.listdir(path_questionnaire_variables):
        if files.startswith('cov'):
            filenum = files.split('_')[2]
            df = pd.read_csv(f'{path_questionnaire_variables}{files}', sep=',', encoding='utf-8')
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            for index, row in df.iterrows():
                if 'terug te veren' in row['definition_nl']:
                    variable_name = row['variable_name'].split(f'{filenum}_')[1]
                    resilience_ids.add(variable_name)
    resilience_id = list(resilience_ids)[0]
    return resilience_id

def results_data(path_questionnaire_results, resilience_id, resilience_path):
    """
    Get the resilience questions and answers out of the questionnaires
    resilience_df: Dataframe with the resilience anwers 
    """
    resilience_df = pd.DataFrame(columns=['project_pseudo_id'])
    for files in os.listdir(path_questionnaire_results):
        if files.startswith('cov'):
            filenum = files.split('_')[2]
            df = pd.read_csv(f'{path_questionnaire_results}{files}', sep=',', encoding='utf-8')
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
           
            df[df.isin(none_value)] = np.nan
            resilience_col = [col for col in df.columns if resilience_id in col]
            if len(resilience_col) > 0:
                df.rename({resilience_col[0]: filenum}, axis=1, inplace=True)
                resilience_df = pd.merge(resilience_df, df[['project_pseudo_id', filenum]], on=['project_pseudo_id'], how='outer')
    # Save file
    resilience_df.to_csv(f"{resilience_path}resilience_without_mean.tsv.gz", sep='\t',
                        encoding='utf-8', compression='gzip', index=False)
    return resilience_df
    
def calculate_resilience(resilience_path, resilience_df):
    """
    Calculate the mean and median of the resilience questions
    """
    resilience_df = pd.read_csv(f"{resilience_path}resilience_without_mean.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
    resilience_df = resilience_df.fillna(-11)
    resilience_df = resilience_df.set_index('project_pseudo_id')
    col_veer = list(resilience_df.columns)
    # Make intergers of values in column
    resilience_df[col_veer] = resilience_df[col_veer].astype(int)
    resilience_df = resilience_df.replace(-11, np.nan)
    resilience_df = resilience_df.reindex(sorted(resilience_df.columns), axis=1)
    # Calculate mean and meadian of the resilience
    resilience_df[f'resilience_mean'] = resilience_df.loc[:,col_veer].mean(axis=1)
    resilience_df[f'resilience_median'] = resilience_df.loc[:,col_veer].median(axis=1)
    # Save file
    resilience_df.to_csv(f"{resilience_path}resilience.tsv.gz", sep='\t',
                        encoding='utf-8', compression='gzip')


def main():
    config = get_config()
    resilience_path = config['resilience']
    path_questionnaire_variables = config['path_questionnaire_variables']
    path_questionnaire_results = config['path_questionnaire_results']
    resilience_id = variables_data(path_questionnaire_variables)
    resilience_df = results_data(path_questionnaire_results, resilience_id, resilience_path)
    calculate_resilience(resilience_path, resilience_df)
    print('DONE')


if __name__ == '__main__':
    main()

