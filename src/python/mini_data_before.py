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
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/COVID_Anne')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

warnings.filterwarnings('ignore')


def calculate_depressive_before(df, depressive, num_quest):
    # Sort the columns (results of depressive)
    depressive = sorted(depressive)
    # Filter dataframe on columns
    df_select = df[['project_pseudo_id'] + depressive]
    # Make lists
    list_1_2 = list()
    list_3 = list()
    list_3_a = list()
    list_3_c = list()
    list_3_other = list()
    # Loop over columns in depressive (sort)
    for i in depressive:
        num = i.split('_q_')[1]
        # Filter on question 1 and 2 
        if num.startswith('1') or num.startswith('2'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0',
                                                                                                              1).replace(
                'nan', np.nan)
            list_1_2.append(i)
        # Filter on question 3
        elif num.startswith('3'):
            list_3.append(i)
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0',
                                                                                                              1).replace(
                'nan', np.nan)
            # There are several 3a and 3c. So these have to be added together separately later.
            if '_a' in num:
                list_3_a.append(i)
            elif '_c' in num:
                if num[-1].isdigit():
                    list_3_c.append(i)
            else:
                list_3_other.append(i)
    # Add the results of questions 1 and 2 together
    df_select[f'before_{num_quest}_sum_mini_a_1_2'] = df_select.loc[:, list_1_2].sum(axis=1)
    # Add the results of questions 3a
    df_select[f'before_{num_quest}_sum_mini_a_3a'] = df_select[list_3_a].max(axis=1)
    # Add the results of questions 3c
    df_select[f'before_{num_quest}_sum_mini_a_3c'] = df_select[list_3_c].max(axis=1)
    # Extend list with 3a and 3c
    list_3_other.extend([f'before_{num_quest}_sum_mini_a_3a', f'before_{num_quest}_sum_mini_a_3c'])
    # Add the results of questions 3
    df_select[f'before_{num_quest}_sum_mini_a_3_all'] = df_select.loc[:, list_3_other].sum(axis=1)
    # Add question 1 and 2 for the all sum.
    list_3_other.extend(list_1_2)
    # Add all questions for depressive together
    df_select[f'before_{num_quest}_sum_mini_a_all'] = df_select.loc[:, list_3_other].sum(axis=1)
    # Return dataframe with only depressive answers
    return df_select


def calculate_anxiety_before(df, anxiety, num_quest):
    # Sort the columns (result of anxiety)
    anxiety = sorted(anxiety)
    # Filter dataframe on columns
    df_select = df[['project_pseudo_id', f'{num_quest}_date'] + anxiety]
    # Make lists
    list_1_ab = list()
    list_other = list()
    list_3 = list()
    # Loop over columns in anxiety (sort)
    for i in anxiety:
        num = i.split('_q_')[1]
        # Filter on question 1
        if num.startswith('1'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0',
                                                                                                              1).replace(
                'nan', np.nan)
            list_1_ab.append(i)
        # Filter on question 2
        elif num.startswith('2'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0',
                                                                                                              1).replace(
                'nan', np.nan)
            df_select[f'before_{num_quest}_sum_mini_o_2'] = df_select.loc[:, [i]].sum(axis=1)
            list_other.append(i)
        # Filter on question 3
        elif num.startswith('3'):
            list_3.append(i)
            list_other.append(i)
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0',
                                                                                                              1).replace(
                'nan', np.nan)
    # Add the results of questions 1a en 1b together
    df_select[f'before_{num_quest}_sum_mini_o_1_ab'] = df_select.loc[:, list_1_ab].sum(axis=1)
    list_other.append(f'before_{num_quest}_sum_mini_o_1_ab')
    # Add the results of questions 3
    df_select[f'before_{num_quest}_sum_mini_o_3_all'] = df_select.loc[:, list_3].sum(axis=1)
    # Add the results of all the questions
    df_select[f'before_{num_quest}_sum_mini_o_all'] = df_select.loc[:, list_other].sum(axis=1)
    # Return dataframe with only anxiety answers
    return df_select


def mini_before_covid(data_QOL_path, mini_path):
    # Empty dataframes
    mini = pd.DataFrame(columns=['project_pseudo_id'])
    dep_all = pd.DataFrame(columns=['project_pseudo_id'])
    aux_all = pd.DataFrame(columns=['project_pseudo_id'])
    name_date_col = ''
    # Loop over different OR (1a, 2a, 3a)
    for num_quest in ['1a', '2a', '3a']:
        # Make lists and sets
        set_type_mini = set()
        # Columns with results of depressive
        depressive = list()
        # Columns with results of anxiety
        anxiety = list()
        # Read dataframe
        df = pd.read_csv(f"{data_QOL_path}QOL_old/df/{num_quest}_mini.tsv.gz", sep='\t', encoding='utf-8',
                         compression='gzip')
        # Create nan values from the following values in the list 
        none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
        df[df.isin(none_value)] = np.nan
        # Loop over columns of df
        for col in df.columns:
            # Check if 'mini' is in column
            if 'mini' in col:
                # Check if the mini question belongs to A (depressive) or O (anxiety)
                type_mini = col.split('_')[2]
                sort_mini = col.split('_')[1]
                set_type_mini.add(type_mini)
                if type_mini == 'a' and sort_mini == 'mini':
                    depressive.append(col)
                elif type_mini == 'o' and sort_mini == 'mini':
                    anxiety.append(col)
            # Check date for later filtering
            if 'date' in col:
                name_date_col = col
                df[name_date_col] = pd.to_datetime(df[name_date_col], errors='coerce')

        # Call calculate_depressive_before
        df_dep = calculate_depressive_before(df, depressive, num_quest)
        # Call calculate_anxiety_before
        df_anx = calculate_anxiety_before(df, anxiety, num_quest)
        # Only filter 3a by date as it may contain answers from during the pandemic as well
        if num_quest == '3a':
            df_date = df[['project_pseudo_id', name_date_col]]
            df_date = df_date[(df_date[name_date_col] < '2020-01')]
            df_date['3a_before_2020'] = 'yes'
        else:
            df_date = df[['project_pseudo_id', name_date_col]]
        # Merge df_date and df_anx
        aux_all = pd.merge(aux_all, df_anx, on=['project_pseudo_id'], how='outer')
        aux_all = pd.merge(df_date, aux_all, on=['project_pseudo_id'], how='outer')
        # Merge df_date and df_dep
        dep_all = pd.merge(dep_all, df_dep, on=['project_pseudo_id'], how='outer')
        dep_all = pd.merge(df_date, dep_all, on=['project_pseudo_id'], how='outer')
        # Merge df_dep, df_aux and df_date
        df_dep_anx = pd.merge(df_dep, df_anx, on=['project_pseudo_id'], how='outer')
        df_dep_anx = pd.merge(df_date, df_dep_anx, on=['project_pseudo_id'], how='outer')

        df_dep_anx.to_csv(f"{mini_path}{num_quest}_filter_mini_dep_aux.tsv.gz", sep='\t',
                          encoding='utf-8', compression='gzip', index=False)
        # Merge all the dep_anx with mini
        mini = pd.merge(mini, df_dep_anx, on=['project_pseudo_id'], how='outer')

    mini.to_csv(f"{mini_path}ALL_filter_mini_dep_aux.tsv.gz", sep='\t',
                encoding='utf-8', compression='gzip', index=False)
    aux_all.to_csv(f"{mini_path}aux_filter_mini.tsv.gz", sep='\t',
                   encoding='utf-8', compression='gzip', index=False)
    dep_all.to_csv(f"{mini_path}dep_filter_mini.tsv.gz", sep='\t',
                   encoding='utf-8', compression='gzip', index=False)
    # Filter on the columns with before
    # Then you have the columns with sum of questions from before the corona
    before_mini = ['project_pseudo_id'] + [col for col in mini.columns if 'before' in col]
    mini[before_mini].to_csv(f"{mini_path}before_mini.tsv.gz", sep='\t',
                             encoding='utf-8', compression='gzip', index=False)
    return mini[before_mini]


def main():
    config = get_config()
    # Different paths
    data_QOL_path = config['data_QOL']
    mini_path = config['MINI']
    # Call mini_before_covid
    before_mini_df = mini_before_covid(data_QOL_path, mini_path)

    print('DONE')


if __name__ == '__main__':
    main()
