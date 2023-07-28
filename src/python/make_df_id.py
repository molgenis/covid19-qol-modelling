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
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

warnings.filterwarnings('ignore')
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config


def calculate_mean_QOL(all_quest, path_save):
    """
    Calculate the mean per date. (=df_q)
    Calculate the mean per date per person (=df_id)
    """
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
    question_15_or_more_path = config['question_15_or_more']
    path_save = config['make_df_id']
    directory = os.fsencode(path_directory)
    # Call different functions
    # all_quest = concat_questionnaires_filter(path_directory, directory)
    df = pd.read_csv(f'{question_15_or_more_path}num_quest_1_filter.tsv.gz', sep='\t',
                     encoding='utf-8', compression='gzip')  # num_quest_1_filter, QOL_data_VL29
    # Select columns
    df = df[['project_pseudo_id', 'responsedate', 'qualityoflife']]
    # Groupby responsedate
    df['size_responsedate'] = df.groupby(['responsedate'])[["responsedate"]].transform('size')
    # df = df[(df['size_participants'] >= 15) & (df['size_responsedate'] >= 50)]
    df = df[df['size_responsedate'] >= 50]
    df_id, df_q = calculate_mean_QOL(df, path_save)
    print('DONE: make_df_id.py')


if __name__ == '__main__':
    main()
