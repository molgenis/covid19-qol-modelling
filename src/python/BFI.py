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


def get_questions(LL_variables):
    """

    """
    df = pd.read_csv(f"{LL_variables}covq_q_t29_variables.csv")
    BFI_list = list()
    for index, row in df.iterrows():
        if 'zie mezelf ' in row['definition_nl']:
            BFI_list.append(row['variable_name'])

    return BFI_list


def check_answers(LL_enumerations, BFI_list):
    """

    """
    df = pd.read_csv(f"{LL_enumerations}covq_q_t29_enumerations.csv")
    for id_BFI in BFI_list:
        df_select = df[df['variable_name'].str.contains(id_BFI)]


def get_results(LL_results, BFI_list, BFI_path):
    """

    """
    column_names = ['project_pseudo_id', 'age', 'gender']
    df = pd.read_csv(f"{LL_results}covq_q_t29_results.csv")
    column_names.extend(BFI_list)
    df_BFI = df[column_names]
    df_BFI.to_csv(f'{BFI_path}BFI.tsv.gz', sep='\t', encoding='utf-8',
                  compression='gzip', index=False)
    return df_BFI


def analyse_BFI(df_BFI, BFI_path):
    """

    """
    column_names = ['project_pseudo_id', 'age', 'gender']
    none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
    df_BFI[df_BFI.isin(none_value)] = np.nan
    reverse_items = ['covt29_personality_adu_q_1_o', 'covt29_personality_adu_q_1_l', 'covt29_personality_adu_q_1_c']
    category_dict = {
        'n': ['covt29_personality_adu_q_1_e', 'covt29_personality_adu_q_1_j', 'covt29_personality_adu_q_1_o'],
        'e': ['covt29_personality_adu_q_1_b', 'covt29_personality_adu_q_1_h', 'covt29_personality_adu_q_1_l'],
        'o': ['covt29_personality_adu_q_1_d', 'covt29_personality_adu_q_1_i', 'covt29_personality_adu_q_1_n'],
        'a': ['covt29_personality_adu_q_1_c', 'covt29_personality_adu_q_1_f', 'covt29_personality_adu_q_1_m'],
        'c': ['covt29_personality_adu_q_1_a', 'covt29_personality_adu_q_1_g', 'covt29_personality_adu_q_1_k']
    }
    for key, value in category_dict.items():
        # Select columns
        df_select = df_BFI[value]
        # Make int of columns
        df_select = df_select.replace(np.nan, -2)
        df_select[value] = df_select[value].astype(int)
        df_select = df_select.replace(-2, np.nan)
        # Check if there is an overlap with value (list from dict) and reverse_items
        if len(list(set(value) & set(reverse_items))) > 0:
            # If there is a reverse column make it another number
            # When it says 5 it becomes 8-5=3 etc.
            overlap_column = list(set(value) & set(reverse_items))[0]
            df_select[f'reverse_{overlap_column}'] = 8 - df_select[overlap_column]

            df_select.drop(overlap_column, axis=1, inplace=True)
            df_select.rename(columns={f'reverse_{overlap_column}': overlap_column}, inplace=True)
        # Sum the values together (this is with the adjusted reverse value)
        df_BFI[f'{key}_sum'] = df_select[value].sum(axis=1)
        column_names.append(f'{key}_sum')
    # Save file
    df_BFI.to_csv(f'{BFI_path}BFI_with_sum.tsv.gz', sep='\t', encoding='utf-8',
                  compression='gzip', index=False)
    df_BFI_sum = df_BFI[column_names]
    df_BFI_sum.to_csv(f'{BFI_path}BFI_only_with_sum.tsv.gz', sep='\t', encoding='utf-8',
                      compression='gzip', index=False)
    return df_BFI, df_BFI_sum


def main():
    config = get_config()
    LL_variables = config['path_questionnaire_variables']
    LL_results = config['path_questionnaire_results']
    LL_enumerations = config['path_questionnaire_enumerations']
    BFI_path = config['BFI']
    BFI_list = get_questions(LL_variables)
    # check_answers(LL_enumerations, BFI_list)
    df_BFI = get_results(LL_results, BFI_list, BFI_path)
    df_BFI, df_BFI_sum = analyse_BFI(df_BFI, BFI_path)

    print('DONE: BFI.py')


if __name__ == '__main__':
    main()
