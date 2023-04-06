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
from scipy import stats
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')


def get_questions(LL_variables):
    df = pd.read_csv(f"{LL_variables}covq_q_t29_variables.csv")
    #print(df)
    BFI = list()
    for index, row in df.iterrows():
        if 'zie mezelf ' in row['definition_nl']:
            #print(f"{row['variable_name']}-{row['definition_nl']}")
            BFI.append(row['variable_name'])
            
    return BFI
    
def check_answers(LL_enumerations, BFI):
    df = pd.read_csv(f"{LL_enumerations}covq_q_t29_enumerations.csv")
    #print(df)
    for id_BFI in BFI:
        df_select = df[df['variable_name'].str.contains(id_BFI)]
        #print(df_select)


def get_results(LL_results, BFI):
    column_names = ['project_pseudo_id', 'age', 'gender']
    df = pd.read_csv(f"{LL_results}covq_q_t29_results.csv")
    column_names.extend(BFI)
    #print(column_names)
    df_BFI = df[column_names]
    #print(df_BFI)
    df_BFI.to_csv('/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/VL/BFI.tsv.gz', sep='\t', encoding='utf-8',
                  compression='gzip', index=False)
    return df_BFI
    
def analyse_BFI(df_BFI):
    column_names = ['project_pseudo_id', 'age', 'gender']
    none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
    df_BFI[df_BFI.isin(none_value)] = np.nan
    reverse_items = ['covt29_personality_adu_q_1_o', 'covt29_personality_adu_q_1_l', 'covt29_personality_adu_q_1_c']
    category_dict = { 'n' : ['covt29_personality_adu_q_1_e', 'covt29_personality_adu_q_1_j', 'covt29_personality_adu_q_1_o'],
                      'e' : ['covt29_personality_adu_q_1_b', 'covt29_personality_adu_q_1_h', 'covt29_personality_adu_q_1_l'],
                      'o' : ['covt29_personality_adu_q_1_d', 'covt29_personality_adu_q_1_i', 'covt29_personality_adu_q_1_n'],
                      'a' : ['covt29_personality_adu_q_1_c', 'covt29_personality_adu_q_1_f', 'covt29_personality_adu_q_1_m'],
                      'c' : ['covt29_personality_adu_q_1_a', 'covt29_personality_adu_q_1_g', 'covt29_personality_adu_q_1_k']
                    }
    for key, value in category_dict.items():
        print(key)
        # Select columns
        df_select = df_BFI[value]
        # Make int of columns
        df_select = df_select.replace(np.nan,-2)
        df_select[value] = df_select[value].astype(int)
        df_select = df_select.replace(-2,np.nan)
        # Check of er een overlap is met value (lijst uit dict) en reverse_items
        if len(list(set(value) & set(reverse_items))) > 0:
            # Wanneer er een reverse column is maak er een ander getal van
            # Wanneer er 5 staat wordt het 8-5=3 etc.
            overlap_column = list(set(value) & set(reverse_items))[0]
            df_select[f'reverse_{overlap_column}'] = 8 - df_select[overlap_column]
            print(df_select[[f'reverse_{overlap_column}', overlap_column]])
            df_select.drop(overlap_column, axis=1, inplace=True)
            df_select.rename(columns={f'reverse_{overlap_column}': overlap_column}, inplace=True)
        #df_select[f'{key}_sum'] = df_select[value].sum(axis=1)
        # Sum de values bij elkaar op (dit is dus met de aangepaste reverse value)
        df_BFI[f'{key}_sum'] = df_select[value].sum(axis=1)
        print(df_select[value])
        print(df_BFI[value + [f'{key}_sum']])
        column_names.append(f'{key}_sum')
        #print(df_select)
        #print('---')
    
    #print(df_BFI)
    df_BFI.to_csv('/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/VL/BFI_with_sum.tsv.gz', sep='\t', encoding='utf-8',
                  compression='gzip', index=False)
    #print(df_BFI[column_names])
    df_BFI_sum = df_BFI[column_names]
    df_BFI_sum.to_csv('/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/VL/BFI_only_with_sum.tsv.gz', sep='\t', encoding='utf-8',
                      compression='gzip', index=False)
    return df_BFI, df_BFI_sum
    
def correlation(df_BFI_sum):
    path_save = '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/with_just_mean/plots/BFI/'
    sum_cols = [col for col in df_BFI_sum.columns if 'sum' in col]
    # print(sum_cols)
    df_sum = df_BFI_sum[sum_cols]
    correlation_sum = df_sum.corr()
    # print(correlation_sum)
    
    path_covariance = f'/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/with_just_mean/QOL_covariance_correlation_beta.tsv.gz'
    covariance_df = pd.read_csv(path_covariance, sep='\t', encoding='utf-8', compression='gzip')
    # Remove 'Unnamed: 0'
    covariance_df = covariance_df.iloc[:, 1:]
    # Select only the people with more then 15 questionnaires
    covariance_df = covariance_df[covariance_df['times'] >= 15]
    # print(covariance_df.columns)
    # print(covariance_df[['variance_x', 'variance_y']])
    
    for val in ['beta', 'correlation', 'variance_x']:
        print(val)
        
        merge_file = pd.merge(df_BFI_sum, covariance_df[['project_pseudo_id', val]], on=['project_pseudo_id'], how='left')
        #TODO
        column_names = ['project_pseudo_id', 'age', 'gender']
        
        merge_file_select = merge_file[sum_cols + [val]]
        correlation_sum_2 = merge_file_select.corr()
        #   print(correlation_sum_2)
        for xx in merge_file.columns:
            if xx != 'project_pseudo_id':
                sns.displot(merge_file[xx])
                # Add title
                plt.title(f"histogram {xx}")
                plt.tight_layout()
                # Save plot
                plt.savefig(f'{path_save}hist_{xx}.png')
                # Clear plot
                plt.clf()
        for col in sum_cols:
            #TODO
            make_model(merge_file, column_names, col, val)
            #   print(col)
            dict_count = dict(Counter(list(merge_file[col])))
            #   print(dict_count)
            plt.bar(list(dict_count.keys()), list(dict_count.values()))
            # labels
            plt.xlabel(col)
            plt.ylabel('number of people')
            # Add title
            plt.title(f"barplot {col}")
            plt.tight_layout()
            # Save plot
            plt.savefig(f'{path_save}bar_{col}.png')
            # Clear plot
            plt.clf()
            
            
            df_sel = merge_file[['project_pseudo_id', col, val]]
            df_sel = df_sel.dropna()
            plt.scatter(df_sel[col], df_sel[val])
            
            
            # pearson correlatie TODO
            pearsonr_values = stats.pearsonr(df_sel[col], df_sel[val])
            #print(pearsonr_values)
            #print(pearsonr_values[1])
            rho, pval = stats.spearmanr(df_sel[col], df_sel[val])
            #print(pval)
            
            # labels
            plt.xlabel(col)
            plt.ylabel(val)
            # Add title
            plt.title(f"scatterplot {col} - {val} \n spearman: {pval}")
            plt.tight_layout()
            # Save plot
            plt.savefig(f'{path_save}scatter_{col}_{val}.png')
            # Clear plot
            plt.clf()

def fit_model_mixedlm(df_sel, col, val):
    # Create Linear Mixed Effect model
    model = smf.mixedlm(
        f"{val} ~ {col} + C(gender) + age",
        data=df_sel, groups=df_sel['project_pseudo_id'])
    print(f'for {val} and {col}')
    # Fit the model
    result = model.fit(method='lbfgs')
    print(result.summary())
    resid = result.resid

    summary_results_top = result.summary().tables[0]
    summary_results_bottom = result.summary().tables[1]

    summary_results_bottom['coef_long'] = result.params
    summary_results_bottom['std_err_long'] = result.bse
    summary_results_bottom['z_values_long'] = result.tvalues
    summary_results_bottom['p_values_long'] = result.pvalues

    return result, summary_results_bottom

def fit_model(df_sel, col, val):
    path_save = '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/with_just_mean/plots/BFI/'
    print(f'for {val} and {col}')
    # print(df_sel)
    df_sel['gender'].replace(['FEMALE','MALE'], [0,1],inplace=True)
    df_sel=df_sel.set_index('project_pseudo_id')
    
    y = df_sel[val].values  #Target Variable
    #transform the data using box-cox: https://towardsdatascience.com/is-normal-distribution-necessary-in-regression-how-to-track-and-fix-it-494105bc50dd
    # sample_transformed_y, lambd = stats.boxcox(y)
    X = df_sel[['age', 'gender', col]].values #Feature Matrix
    X = StandardScaler().fit_transform(X)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print("train and test data are split:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    #---Linear Regression-----------------------------------------
    line_reg=LinearRegression()
    line_reg.fit(X_train,y_train)
    print('Intercept: ', line_reg.intercept_)
    print('Coefficients: ', line_reg.coef_)

    #Predicting the y_value of linear regression
    lin_reg_y_predicted = line_reg.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': lin_reg_y_predicted})
    results["residuals"] = results["Actual"] - results["Predicted"]
    print(results)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    # ax = sm.qqplot(results["residuals"])
    ax = plt.scatter(np.arange(0, len(results["residuals"])), results['residuals'], s=7, alpha=0.1, linewidth=0)
    print("\nJ-B:")
    print(stats.jarque_bera(results['residuals']))
    print(stats.jarque_bera(results['residuals'])[1])

    
    #plt.scatter(results["residuals"], results["Predicted"])
    # Save plot
    plt.savefig(f'{path_save}residuals_{col}_{val}.png')
    # Clear plot
    plt.clf()
    
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    print(regr.predict(X_test[:2]))
    print(regr.score(X_test, y_test))

    # #calculating the rmse 
    # lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_y_predicted))
    # print('lin_reg_rmse : ', lin_reg_rmse)

    # #Linear Regression Accuracy with test set
    # lin_reg_acc = r2_score(y_test, lin_reg_y_predicted)
    # print('Linear Regression R2: ',lin_reg_acc)
    # actual_minus_predicted = sum((y_test - lin_reg_y_predicted)**2)
    # actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
    # r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
    # print('R2:', r2)

def make_model(merge_file, column_names, col, val):
    print('=====================================')
    list_sel_col = column_names + [col, val]
    df_sel = merge_file[list_sel_col]
    df_sel.dropna(inplace=True)
    # result, summary_results_bottom = fit_model(df_sel, col, val)
    fit_model(df_sel, col, val)
    


def main():
    config = get_config()
    my_folder = config['my_folder']
    LL_variables = config['path_questionnaire_variables']
    LL_results = config['path_questionnaire_results']
    LL_enumerations = config['path_questionnaire_enumerations']
    BFI = get_questions(LL_variables)
    #check_answers(LL_enumerations, BFI)
    df_BFI = get_results(LL_results, BFI)
    df_BFI, df_BFI_sum = analyse_BFI(df_BFI)
    # correlation(df_BFI_sum)

    print('DONE')


if __name__ == '__main__':
    main()



