# -*- coding: utf-8 -*-
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
warnings.filterwarnings('ignore')
import scipy.stats as stats
from statistics import median, mean


def calculate_U(group_a, group_b, name_a, name_b):
    """
    
    Calculate U
    
    """
    #http://users.sussex.ac.uk/~grahamh/RM1web/MannWhitneyHandout%202011.pdf
    a_b = list(group_a) + list(group_b)
    a_b_values = list([0] * len(group_a)) + list([1] * len(group_b))
    df = pd.DataFrame(data = {'beta': a_b, 'group': a_b_values})
    # Step 1
    df['average_rank'] = df['beta'].rank(method='average')
    # Step 2
    T1 = sum(df[df['group'] == 0]['average_rank'])
    # Step 3
    T2 = sum(df[df['group'] == 1]['average_rank'])
    # Step 4
    TX = max(T1, T2)
    # Step 5
    N1 = len(group_a)
    N2 = len(group_b)
    if TX == T1:
        NX = N1
    else:
        NX = N2
    # Step 6
    U = N1 * N2 + NX * (NX + 1) / 2 - TX
    # wikipedia: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    U1 = T1 - (N1 * (N1 + 1)) / 2
    U2 = T2 - (N2 * (N2 + 1)) / 2
    print(f'T1:{T1} - T2:{T2} - TX:{TX}')
    print(f'N1:{N1} - N2:{N2} - NX:{NX}')
    print(f'U:{U}')
    print(f'U1:{U1} - U2:{U2}')
    if U1 > U2:
        print(f'U1 ({name_a}) > U2 ({name_b})')
    else:
        print(f'U1 ({name_a}) < U2 ({name_b})')
    print(f'{name_a} - median:{median(list(group_a))}, U1:{U1}, mean:{mean(list(group_a))}, len: {len(group_a)}')
    print(f'{name_b} - median:{median(list(group_b))}, U2:{U2}, mean:{mean(list(group_b))}, len: {len(group_b)}')
    print()

def calculate_group(df_QOL_select, column_group):
    """
    
    """
    female_df = df_QOL_select[df_QOL_select['gender'] == 'FEMALE']
    male_df = df_QOL_select[df_QOL_select['gender'] != 'FEMALE']
    print(column_group)
    print(f'female: {len(female_df)} - mean_age: {female_df["mean_age"].mean()}')
    print(f'male: {len(male_df)} - mean_age: {male_df["mean_age"].mean()}')

def select_columns(df_QOL, variable, column_group):
    """
    
    """
    df_QOL = df_QOL[df_QOL['mean_age'].notna()]
    df_QOL = df_QOL[df_QOL['gender'].notna()]
    df_QOL_select = df_QOL[['project_pseudo_id', column_group, variable]]
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    return df_QOL_select

def spearman_test(df_QOL_select, variable, column_group):
    """
    
    """
    spearman_values = stats.spearmanr(df_QOL_select[column_group], df_QOL_select[variable])
    print(f'-----{column_group} - spearmanr')
    print(spearman_values)
    print(spearman_values.pvalue)

def wilcoxon_U_test(first_group, second_group, column_group):
    """
    
    """
    # wilcoxon_values = stats.ranksums(first_group, second_group)
    wilcoxonU = stats.mannwhitneyu(first_group, second_group)
    print(f'-----{column_group} - mannwhitneyu')
    print(wilcoxonU)
    print(wilcoxonU.pvalue)




