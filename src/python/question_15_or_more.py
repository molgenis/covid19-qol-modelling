#!/usr/bin/env python3

# ---------------------------------------------------------
# Author: Anne van Ewijk
# University Medical Center Groningen / Department of Genetics
#
# Copyright (c) Anne van Ewijk, 2023
#
# ---------------------------------------------------------

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
import warnings

warnings.filterwarnings('ignore')


def get_data(path_directory, question_15_or_more_path):
    """
    Check whether participants have completed 15 or more questionnaires
    """
    # Create empty dataframe with column names
    df_quest = pd.DataFrame(
        columns=['project_pseudo_id', 'responsedate', 'gender', 'age', 'qualityoflife', 'num_quest'])
    # Loop over files in path_directory
    for files in os.listdir(path_directory):
        if files.startswith('covq'):
            filenum = files.split('_')[2]
            filenum = str(filenum.replace('t', ''))
            # QOL question does not appear in the first questionnaire
            if filenum != '01':
                df = pd.read_csv(f'{path_directory}{files}', sep=',', encoding='utf-8')
                for col in df.columns:
                    if '_qualityoflife_' in col:
                        qual_col = col
                df = df[['project_pseudo_id', f'covt{filenum}_responsedate_adu_q_1', 'gender', 'age', qual_col]]
                df.rename(columns={f'covt{filenum}_responsedate_adu_q_1': 'responsedate'}, inplace=True)
                df.rename(columns={qual_col: 'qualityoflife'}, inplace=True)
                none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
                df[df.isin(none_value)] = np.nan
                df['num_quest'] = filenum
                df_quest = pd.merge(df_quest, df, how='outer',
                                    on=['project_pseudo_id', 'responsedate', 'gender', 'age', 'qualityoflife',
                                        'num_quest'])
    # Save file
    df_quest.to_csv(f'{question_15_or_more_path}num_quest_1.tsv.gz', sep='\t', encoding='utf-8',
                    compression='gzip', index=False)
    # Groupby project_pseudo_id
    df = df_quest.groupby('project_pseudo_id').size().reset_index()
    df.rename(columns={df.columns[1]: "times_part"}, inplace=True)
    # Filter on times_part
    df = df[df['times_part'] >= 15]
    df.to_csv(f'{question_15_or_more_path}num_quest_1_15_or_more.tsv.gz', sep='\t', encoding='utf-8',
              compression='gzip', index=False)
    # Merge dataframe
    df_merge = pd.merge(df, df_quest, how='left', on=['project_pseudo_id'])
    df_merge.to_csv(f'{question_15_or_more_path}num_quest_1_filter.tsv.gz', sep='\t', encoding='utf-8',
                    compression='gzip', index=False)


def main():
    config = get_config()
    path_directory = config['path_questionnaire_results']
    question_15_or_more_path = config['question_15_or_more']
    get_data(path_directory, question_15_or_more_path)
    print('DONE: question_15_or_more.py')


if __name__ == '__main__':
    main()
