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
import re

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/COVID_Anne')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

warnings.filterwarnings('ignore')


def above_value(mini_df, cols, num):
    # Select how often 1 occurs
    mini_df[f'between_mini_{num}_1'] = mini_df[cols].sum(axis=1)
    # Select how often 0 occurs
    mini_df[f'between_mini_{num}_0'] = (mini_df[cols] == 0).sum(axis=1)
    # Calculate how often 1 is entered in percentages
    mini_df[f'between_mini_{num}_percent_1'] = mini_df[f'between_mini_{num}_1'] / (
            mini_df[f'between_mini_{num}_1'] + mini_df[f'between_mini_{num}_0']) * 100
    # Filter on people who have filled in 50% on more 1
    mini_df[f'between_above_mini_{num}'] = np.where(mini_df[f'between_mini_{num}_percent_1'] >= 50, 1, 0)


def sum_same_quest(mini_df, list_cat, list_fatigue):
    """

    Selects everyone who has answered yes to the mini question 50% or more times
    """
    list_cat = sorted(list_cat)
    for num in list_cat:
        if 'fatigue_adu_' not in num:
            mini_col = [col for col in mini_df.columns if f'mini{num}' in col]
            mini_df[mini_col] = mini_df[mini_col].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace(
                '1.0', 1).replace('nan', np.nan)
            above_value(mini_df, mini_col, num)

    # {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
    # 1 = Yes, that is absolutely correct
    # 7 = No, that is not correct

    if list_fatigue != '':
        for col_fat in list_fatigue:
            mini_df[col_fat] = mini_df[col_fat].astype(str).replace('6', 0).replace('6.0', 0).replace('5', 0).replace(
                '5.0', 0).replace('4', 0).replace('4.0', 0).replace('3', 1).replace('3.0', 1).replace('2', 1).replace(
                '2.0', 1).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)

        frag_a = [col for col in list_fatigue if col.endswith('a')]
        frag_b = [col for col in list_fatigue if col.endswith('b')]
        frag_d = [col for col in list_fatigue if col.endswith('d')]
        above_value(mini_df, frag_a, f'o3c_a')
        above_value(mini_df, frag_b, f'o3c_b')
        above_value(mini_df, frag_d, f'o3c_d')
        # Select how often 1 occurs
        mini_df[f'between_above_mini_o3c'] = mini_df[
            ['between_above_mini_o3c_a', 'between_above_mini_o3c_b', 'between_above_mini_o3c_d']].max(axis=1)
        mini_df.drop(['between_above_mini_o3c_a', 'between_above_mini_o3c_b', 'between_above_mini_o3c_d'], axis=1,
                     inplace=True)

    mini_col_above = ['project_pseudo_id'] + [col for col in mini_df.columns if f'between_above_mini_' in col]
    mini_above = mini_df[mini_col_above]
    return mini_above


def calculate_depressive_between(mini_df, depressive):
    mini_above = sum_same_quest(mini_df, depressive, '')
    # Select on questions
    list_df_3 = [col for col in mini_above.columns if f'mini_a3' in col]
    list_1_2 = list(set(list(mini_above.columns)) - set(list_df_3))
    # Add the results of questions 1 and 2 together
    mini_above[f'between_sum_mini_a_1_2'] = mini_above.loc[:, list_1_2].sum(axis=1)
    # Add the results of questions 3
    mini_above[f'between_sum_mini_a_3_all'] = mini_above.loc[:, list_df_3].sum(axis=1)
    # Add all questions for depressive together
    mini_above[f'between_sum_mini_a_all'] = mini_above.loc[:, list_df_3 + list_1_2].sum(axis=1)
    sum_col = ['project_pseudo_id'] + [col for col in mini_above.columns if f'between_sum_mini_a_' in col]
    return mini_above[sum_col]


def calculate_anxiety_between(mini_df, anxiety):
    # a3b = o3f
    # a3f = o3d
    # fatigue = o3c
    #   1_a/2_a = I felt tired
    #   1_b/2_b = I was easily tired
    #   1_d/2_d = I felt physically exhausted
    list_3b = [col for col in mini_df.columns if f'minia3b' in col]
    list_3f = [col for col in mini_df.columns if f'minia3f' in col]
    list_fatigue = [col for col in mini_df.columns if f'fatigue' in col]

    for value in list_3b + list_3f:
        value = value.split('_')[1].replace('mini', '')
        anxiety.add(value)

    mini_above = sum_same_quest(mini_df, anxiety, list_fatigue)
    mini_above.rename(columns={'between_above_mini_a3b': 'between_above_mini_o3f',
                               'between_above_mini_a3f': 'between_above_mini_o3d'}, inplace=True)

    list_1_ab = [col for col in mini_above.columns if f'_mini_o1' in col]
    list_2 = [col for col in mini_above.columns if f'mini_o2' in col]
    list_3 = [col for col in mini_above.columns if f'mini_o3' in col]

    # Add the results of questions 1
    mini_above[f'between_sum_mini_o_1ab'] = mini_above.loc[:, list_1_ab].sum(axis=1)
    # Add the results of questions 2
    mini_above[f'between_sum_mini_o_2'] = mini_above.loc[:, list_2].sum(axis=1)
    # Add the results of questions 3
    mini_above[f'between_sum_mini_o_3_all'] = mini_above.loc[:, list_3].sum(axis=1)
    sum_col = ['project_pseudo_id'] + [col for col in mini_above.columns if f'between_sum_mini_o' in col]

    return mini_above[sum_col]


def make_mini_df_between(path_results, mini_path):
    # Empty sets and dataframes
    set_participants = set()
    set_cols = set()
    mini_df = pd.DataFrame(columns=['project_pseudo_id'])
    # Loop over questionnaire results
    for files in os.listdir(path_results):
        # If file starts with 'cov'
        if files.startswith('covq'):
            filenum = files.split('_')[2]
            # Read dataframe
            df = pd.read_csv(f'{path_results}{files}', sep=',', encoding='utf-8')
            # Create nan values from the following values in the list
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            # Update set of participants
            set_participants.update(list(df['project_pseudo_id']))
            # Get columns
            mini_col = [col for col in df.columns if 'mini' in col]
            fatigue = [col for col in df.columns if re.match(r'.*_fatigue_adu_q_[12]_[bad]', col)]
            # Merge the selected dataframe to mini_df
            mini_df = pd.merge(mini_df, df[['project_pseudo_id'] + mini_col + fatigue], on=['project_pseudo_id'],
                               how='outer')
            # Make set of columns names
            for i in list(mini_col + fatigue):
                # Replace covt{num} with ''
                set_cols.add(i.replace(f'cov{filenum}', ''))
    mini_df.to_csv(f"{mini_path}between_mini.tsv.gz", sep='\t',
                   encoding='utf-8', compression='gzip', index=False)
    return mini_df


def mini_between_covid(mini_path, mini_df):
    if mini_df.empty:
        mini_df = pd.read_csv(f"{mini_path}between_mini.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
    set_participants = set(mini_df['project_pseudo_id'])
    set_cols = set()
    for col in mini_df.columns:
        if re.match(r'.*_fatigue_adu_q_[12]_[bad]', col):
            # Replace covt{num} with ''
            set_cols.add(re.sub(r"covt.*_f", "f", col))
        if 'mini' in col:
            # Replace covt{num} with ''
            set_cols.add(re.sub(r"covt.*_m", "m", col))  # covt\d*_

    # Empty sets and lists
    set_type_mini = set()
    depressive = list()
    depressive_set = set()
    anxiety = list()
    anxiety_set = set()
    # Loop over columns mini_df
    for col in mini_df.columns:
        # Check if 'mini' is in column
        if 'mini' in col:
            type_mini = col.split('mini')[1].split('_')[0]
            set_type_mini.add(type_mini)
            # Check if 'a' in column Depressive
            if re.match(r'^a.*', type_mini):
                depressive.append(col)
                depressive_set.add(type_mini)
            # Check if 'o' in column Anxiety
            elif re.match(r'^o.*', type_mini):
                anxiety.append(col)
                anxiety_set.add(type_mini)
        # Check if 'fatigue' is in column
        if 'fatigue' in col:
            fatigue = f"f{col.split('_f')[1]}"
            anxiety.append(col)
            anxiety_set.add(fatigue)

    df_dep = calculate_depressive_between(mini_df, list(depressive_set))
    df_anx = calculate_anxiety_between(mini_df, anxiety_set)
    # # calculate_anxiety_before(mini_df, anxiety, num_quest)
    return mini_df, list(set_participants), df_dep, df_anx


def main():
    config = get_config()
    # Different paths
    mini_path = config['MINI']
    path_results = config['path_questionnaire_results']
    mini_df = pd.DataFrame()

    # Call functions
    mini_df = make_mini_df_between(path_results, mini_path)
    mini_between_covid(mini_path, mini_df)

    print('DONE')


if __name__ == '__main__':
    main()
