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
from statsmodels.stats.weightstats import ttest_ind
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statistics import median, mean, stdev
from collections import Counter

from tests_make_data import get_data_ready, merge_other_data
from test_spearman_wilcoxonU import calculate_U, calculate_group, select_columns, spearman_test, wilcoxon_U_test


def age(df_QOL, variable):
    """
    
    """
    column_group = 'mean_age'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    spearman_test(df_QOL_select, variable, column_group)


def gender(df_QOL, variable):
    """
    
    """
    column_group = 'gender'
    df_QOL_select = select_columns(df_QOL, variable, column_group)    
    beta_female = df_QOL_select[df_QOL_select[column_group] == 'FEMALE'][variable]
    beta_male = df_QOL_select[df_QOL_select[column_group] != 'FEMALE'][variable]
    
    wilcoxon_U_test(beta_female, beta_male, column_group)

    # calculate_U(beta_female, beta_male, 'Female', 'Male')
    # calculate_group(df_QOL_select[df_QOL_select['gender'] == 'FEMALE'], column_group)
    # calculate_group(df_QOL_select[df_QOL_select['gender'] != 'FEMALE'], column_group)
    

def household(df_QOL, variable):
    """
    
    """
    df_QOL_select = df_QOL[['project_pseudo_id', 'mean_age', 'gender', 'household_status', variable]]
    df_QOL_select = df_QOL_select[df_QOL_select['household_status'].notna()]
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    
    for cat_household in list(set(df_QOL_select['household_status'])):
        beta_cat = df_QOL_select[df_QOL_select['household_status'] == cat_household][variable]
        beta_other = df_QOL_select[df_QOL_select['household_status'] != cat_household][variable]
        
        wilcoxon_U_test(beta_cat, beta_other, cat_household)

        # calculate_U(beta_cat, beta_other, cat_household, 'other')
        # calculate_group(df_QOL_select[df_QOL_select['household_status'] == cat_household], cat_household)
        # calculate_group(df_QOL_select[df_QOL_select['household_status'] != cat_household], 'other')


def media(df_QOL, variable):
    """
    
    """
    media_cols = [col for col in df_QOL.columns if 'mediacategory' in col]
    
    for cat_media in media_cols:
        df_QOL_select = df_QOL[['project_pseudo_id', 'mean_age', 'gender', cat_media, variable]]
        df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
        
        beta_False = df_QOL_select[df_QOL_select[cat_media] == False][variable]
        beta_True = df_QOL_select[df_QOL_select[cat_media] != False][variable]

        wilcoxon_U_test(beta_False, beta_True, cat_media)

        # calculate_U(beta_False, beta_True, 'False', 'True')
        # calculate_group(df_QOL_select[df_QOL_select[cat_media] != False], cat_media)
        # calculate_group(df_QOL_select[df_QOL_select[cat_media] == False], 'other')
        

def general_health(df_QOL, variable):
    """
    
    """
    column_group = 'general_health'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    gen_health = {'poor': 1,'mediocre': 2, 'good': 3, 'very good': 4, 'excellent': 5 }
    df_QOL_select[column_group] = df_QOL_select[column_group].map(gen_health)
    spearman_test(df_QOL_select, variable, column_group)


def education(df_QOL, variable):
    """
    
    """
    column_group = 'education'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    educat = {1.0:1, 2.0:7, 3.0:10, 4.0:10, 5.0:15, 6.0:13, 7.0:22, 8.0:22, 9.0:np.nan}
    df_QOL_select[column_group] = df_QOL_select[column_group].map(educat)
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    spearman_test(df_QOL_select, variable, column_group)


def income(df_QOL, variable):
    """
    
    """
    column_group = 'income'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    
    dict_ans = {"<\u20AC500":1.0, "\u20AC501-\u20AC1000":2.0, "\u20AC1001-\u20AC1500":3.0, "\u20AC1501-\u20AC2000":4.0, 
                "\u20AC2001-\u20AC2500":5.0, "\u20AC2501-\u20AC3000":6.0, "\u20AC3001-\u20AC3500":7.0, 
                "\u20AC3501-\u20AC4000":8.0, "\u20AC4001-\u20AC4500":9.0, "\u20AC4501-\u20AC5000":10.0, 
                "\u20AC5001-\u20AC7500":11.0, ">\u20AC7500":12.0, "I prefer not to say":13.0}
    df_QOL_select[column_group] = df_QOL_select[column_group].map(dict_ans)
    df_QOL_select[column_group].replace(13, np.NaN, inplace=True)
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    
    spearman_test(df_QOL_select, variable, column_group)
    

def mini_dep(df_QOL, variable):
    """
    
    """
    column_group_dep = 'major_depressive_episode'
    
    df_QOL_select_dep = df_QOL[['project_pseudo_id', 'mean_age', 'gender', column_group_dep, variable]]
    # df_QOL_select_dep = df_QOL_select_dep.dropna().reset_index(drop=True)
    df_QOL_select_dep = df_QOL_select_dep[df_QOL_select_dep[variable].notna()]
    df_QOL_select_dep[column_group_dep] = df_QOL_select_dep[column_group_dep].fillna(0)
    beta_depression = df_QOL_select_dep[df_QOL_select_dep[column_group_dep] == 1][variable]
    beta_notdep = df_QOL_select_dep[df_QOL_select_dep[column_group_dep] != 1][variable]

    wilcoxon_U_test(beta_depression, beta_notdep, column_group_dep)

    # calculate_U(beta_depression, beta_notdep, 'dep', 'NOT dep')
    # calculate_group(df_QOL_select_dep[df_QOL_select_dep[column_group_dep] == 1], column_group_dep)
    # calculate_group(df_QOL_select_dep[df_QOL_select_dep[column_group_dep] != 1], 'Not dep')



def mini_anx(df_QOL, variable):
    """
    
    """
    column_group_anx = 'generalized_anxiety_disorder'
    df_QOL_select_anx = df_QOL[['project_pseudo_id', 'mean_age', 'gender', column_group_anx, variable]]
    # df_QOL_select_anx = df_QOL_select_anx.dropna().reset_index(drop=True)
    df_QOL_select_anx = df_QOL_select_anx[df_QOL_select_anx[variable].notna()]
    df_QOL_select_anx[column_group_anx] = df_QOL_select_anx[column_group_anx].fillna(0)
    beta_anx = df_QOL_select_anx[df_QOL_select_anx[column_group_anx] == 1][variable]
    beta_notanx = df_QOL_select_anx[df_QOL_select_anx[column_group_anx] != 1][variable]

    wilcoxon_U_test(beta_anx, beta_notanx, column_group_anx)

    # calculate_U(beta_anx, beta_notanx, 'anx', 'NOT anx')
    # calculate_group(df_QOL_select_anx[df_QOL_select_anx[column_group_anx] == 1], column_group_anx)
    # calculate_group(df_QOL_select_anx[df_QOL_select_anx[column_group_anx] != 1], 'Not anx')


def BFI(df_QOL, variable):
    """
    
    """
    BFI_col = [col for col in df_QOL.columns if '_sum' in col]
    for col in BFI_col:
        if 'before_2a' not in col and 'between' not in col:
            df_QOL_select = df_QOL[['project_pseudo_id', col, variable]]
            df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
            spearman_test(df_QOL_select, variable, col)
            

def resilience(df_QOL, variable):
    """
    
    """
    column_group = 'resilience_median'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    spearman_test(df_QOL_select, variable, column_group)
  


def main():
    config = get_config()
    tests_over_groups_and_beta_path = config['tests_over_groups_and_beta']
    question_15_or_more_path = config['question_15_or_more']
    BFI_path = config['BFI']
    create_file_with_groups_path = config['create_file_with_groups']
    calculate_beta_path = config['calculate_beta']
    resilience_path = config['resilience']
    mini_path = config['MINI']
    path_read_QOL = config['path_read_QOL']
    head_top_null_path = config['head_top_null']

    df_QOL = pd.DataFrame()
    
    variable = 'beta'
    df_QOL = get_data_ready(path_read_QOL, tests_over_groups_and_beta_path, calculate_beta_path, create_file_with_groups_path, question_15_or_more_path, BFI_path)
    df_QOL_select = merge_other_data(df_QOL, tests_over_groups_and_beta_path, resilience_path, mini_path, head_top_null_path)   

     
    # df_QOL_filter['beta_abs'] = df_QOL_filter['beta'].abs()
    # variable = 'beta_abs'
    age(df_QOL_select, variable)
    df_QOL_select = df_QOL_select.drop(['responsedate', 'qualityoflife', 'age', 'num_quest'], axis=1)
    df_QOL_select.drop_duplicates(inplace=True)
    general_health(df_QOL_select, variable)
    education(df_QOL_select, variable)
    income(df_QOL_select, variable)
    BFI(df_QOL_select, variable)
    resilience(df_QOL_select, variable)
    gender(df_QOL_select, variable)
    household(df_QOL_select, variable)
    media(df_QOL_select, variable)    
    mini_dep(df_QOL_select, variable)
    mini_anx(df_QOL_select, variable)
    
    
    
    print('DONE')


if __name__ == '__main__':
    main()

