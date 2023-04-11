#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import sys
import re
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/COVID_Anne')
from config import get_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def select_label(before_mini_df, set_participants):
    # Filter dataframe on covid participants
    before_mini_df = before_mini_df[before_mini_df['project_pseudo_id'].isin(set_participants)]
    # A. MAJOR DEPRESSIVE EPISODE
    df_1_2 = before_mini_df[before_mini_df['before_2a_sum_mini_a_1_2'] >= 1]
    major_depressive_episode = df_1_2[df_1_2['before_2a_sum_mini_a_all'] >= 5]
    major_depressive_episode['major_depressive_episode'] = 1
    
    # not_major_depressive_episode = before_mini_df[~before_mini_df['project_pseudo_id'].isin(list(major_depressive_episode['project_pseudo_id']))] #df_1_2[df_1_2['before_2a_sum_mini_a_all'] < 5]
    # not_major_depressive_episode['not_major_depressive_episode'] = 1
    # not_major_depressive_episode['not_major_depressive_episode'] = not_major_depressive_episode['not_major_depressive_episode'].fillna(0)
    # print(dict(Counter(list(not_major_depressive_episode['not_major_depressive_episode']))))

    print('A. MAJOR DEPRESSIVE EPISODE')
    print(len(before_mini_df))
    print(len(df_1_2))
    print(len(major_depressive_episode))
    # print(len(not_major_depressive_episode))
    print()
    # O. GENERALIZED ANXIETY DISORDER
    df_1 = before_mini_df[before_mini_df['before_2a_sum_mini_o_1_ab'] >= 2]
    df_2 = df_1[df_1['before_2a_sum_mini_o_2'] >= 1]
    generalized_anxiety_disorder = df_2[df_2['before_2a_sum_mini_o_3_all'] >= 3]
    generalized_anxiety_disorder['generalized_anxiety_disorder'] = 1
    # not_generalized_anxiety_disorder = before_mini_df[~before_mini_df['project_pseudo_id'].isin(list(generalized_anxiety_disorder['project_pseudo_id']))] #df_2[df_2['before_2a_sum_mini_o_3_all'] < 3]
    # not_generalized_anxiety_disorder['not_generalized_anxiety_disorder'] = 1
    print('O. GENERALIZED ANXIETY DISORDER')
    print(len(before_mini_df))
    print(len(df_1))
    print(len(df_2))
    print(len(generalized_anxiety_disorder))
    # print(len(not_generalized_anxiety_disorder))
    print()

    before_mini_df = pd.merge(before_mini_df, major_depressive_episode[['project_pseudo_id', 'major_depressive_episode']], on=['project_pseudo_id'], how='outer')
    before_mini_df['major_depressive_episode'] = before_mini_df['major_depressive_episode'].fillna(0)
    print(dict(Counter(list(before_mini_df['major_depressive_episode']))))
    # before_mini_df = pd.merge(before_mini_df, not_major_depressive_episode[['project_pseudo_id', 'not_major_depressive_episode']], on=['project_pseudo_id'], how='outer')
    before_mini_df = pd.merge(before_mini_df, generalized_anxiety_disorder[['project_pseudo_id', 'generalized_anxiety_disorder']], on=['project_pseudo_id'], how='outer')
    before_mini_df['generalized_anxiety_disorder'] = before_mini_df['generalized_anxiety_disorder'].fillna(0)
    print(dict(Counter(list(before_mini_df['generalized_anxiety_disorder']))))
    # before_mini_df = pd.merge(before_mini_df, not_generalized_anxiety_disorder[['project_pseudo_id', 'not_generalized_anxiety_disorder']], on=['project_pseudo_id'], how='outer')
    return before_mini_df #major_depressive_episode, not_major_depressive_episode, generalized_anxiety_disorder, not_generalized_anxiety_disorder


def main():
    config = get_config()
    # Different paths
    my_folder = config['my_folder']
    mini_path = config['MINI']
    path_myfolder = config['path_read_QOL']
    path_variables = config['path_questionnaire_variables']
    path_results = config['path_questionnaire_results']
    path_enumerations = config['path_questionnaire_enumerations']
    

    
    
    print('DONE')


if __name__ == '__main__':
    main()

