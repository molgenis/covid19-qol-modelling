#!/usr/bin/env python3

# ---------------------------------------------------------
# Author: Anne van Ewijk
# University Medical Center Groningen / Department of Genetics
#
# Copyright (c) Anne van Ewijk, 2023
#
# ---------------------------------------------------------

# Imports
import sys
import pandas as pd

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

from scipy import stats

warnings.filterwarnings('ignore')


def calculate_cor(df_corr, create_model):
    """
    Calculate different correlations
    """
    # List of columns
    list_col = ['Wind gust', 'avg_temp', 'Minimum Temperature (\u00B0C)', 'Maximum Temperature (\u00B0C)',
                'Sunshine Duration(hour)', 'Maximum Potential Sunshine Duration (%)', 'Precipitation Duration(hour)',
                'Maximum Precipitation Amount(hourly/mm)', 'Maximum Humidity(%)', 'Minimum Humidity(%)',
                'daily_hospitalization', 'new_cases', 'new_deaths', 'daily_hospitalization_2', 'new_tests',
                'people_vaccinated', 'stringency_index', '1day_rolling_avg_temp', '2day_rolling_avg_temp',
                '3day_rolling_avg_temp', '4day_rolling_avg_temp', '5day_rolling_avg_temp', '6day_rolling_avg_temp',
                '7day_rolling_avg_temp', '8day_rolling_avg_temp', '9day_rolling_avg_temp', '10day_rolling_avg_temp',
                '11day_rolling_avg_temp', '12day_rolling_avg_temp', '13day_rolling_avg_temp', '14day_rolling_avg_temp',
                '15day_rolling_avg_temp', '16day_rolling_avg_temp', '17day_rolling_avg_temp', '18day_rolling_avg_temp',
                '19day_rolling_avg_temp', '20day_rolling_avg_temp', '21day_rolling_avg_temp', 'daylight_hours',
                '1day_daylight_hours', '2day_daylight_hours', '3day_daylight_hours', '4day_daylight_hours',
                '5day_daylight_hours', '6day_daylight_hours', '7day_daylight_hours', '8day_daylight_hours',
                '9day_daylight_hours', '10day_daylight_hours', '11day_daylight_hours', '12day_daylight_hours',
                '13day_daylight_hours', '14day_daylight_hours', '15day_daylight_hours', '16day_daylight_hours',
                '17day_daylight_hours', '18day_daylight_hours', '19day_daylight_hours', 'Open', 'High', 'Low', 'Close',
                'Adj Close', 'Volume', 'Consumer_trust', 'Economic_climate', 'Buy_willingness', 'AnnualRateOfChange',
                'AnnualRateOfChangeDerived', 'News Sentiment']
    # Open en write file
    myfile = open(f'{create_model}spearman_table2.tsv', 'w')
    # Write header
    myfile.writelines('variable\trho\tpval\n')
    # Loop over list of columns
    for col in list_col:
        # Select two columns
        sel_col = df_corr[['qualityoflife', col]]
        # QOL must not contain nan values
        sel_col = sel_col[sel_col['qualityoflife'].notna()]
        # 'col' must not contain nan values
        sel_col = sel_col[sel_col[col].notna()]
        # Calculate spearman correlation
        rho, pval = stats.spearmanr(sel_col[col], sel_col['qualityoflife'])
        # Write to file
        myfile.writelines(f"{col}\t{rho}\t{pval}\n")
    myfile.close()

def corr_hosp_death(df_corr, create_model_path):
    df = pd.read_csv(f'{create_model_path}predicted_qual_linear_regression.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    df_merge = pd.merge(df_corr[['date', 'qualityoflife', 'new_deaths', 'daily_hospitalization']], df[['date', 'residuals']], how="outer", on=["date"])

    for j in ['residuals', 'qualityoflife']:
        print()
        for i in ['new_deaths', 'daily_hospitalization']:
            df_select = df_merge[['date', i, j]]
            df_select.dropna(inplace=True)
            # Calculate spearman correlation
            rho, pval = stats.spearmanr(df_select[i], df_select[j])
            print(f'{j}:{i} - rho: {rho} - pval {pval}\n')
