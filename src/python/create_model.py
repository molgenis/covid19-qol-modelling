#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import seaborn as sns
import warnings
from collections import Counter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as smf
import datetime as dt
import scipy.stats as stats
from scipy import stats
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')
from make_dataframe_for_correlation import add_weather_QOL, add_hospitalization, add_stringency_index, sunrise_sunset, add_other_cat
from correlation_for_table2 import calculate_cor




def predict_values(qol_mod, qol_df, reg_model, reg, X, X_total, type_model, create_model):
    """
    Predict the QOL

    """
    #predict the whole dataset
    if reg == '':
        qol_mod['y_pred']=reg_model.predict(X)
        qol_df['y_pred']=reg_model.predict(X_total)
    else:
        qol_mod['y_pred']=reg_model.predict(reg.fit_transform(X))
        qol_df['y_pred']=reg_model.predict(reg.fit_transform(X_total))


    qol_df = qol_df[qol_df['date'] >= min(qol_mod['date'])]

    qol_mod.to_csv(f'{create_model}average_points_{type_model}.tsv.gz', sep='\t', encoding='utf-8', #, #model_QOLpoint3
                           compression='gzip')
    qol_df.to_csv(f'{create_model}predicted_points_{type_model}.tsv.gz', sep='\t', encoding='utf-8', # ,model_otherpoints3
                           compression='gzip')
    qol_df.to_csv(f'{create_model}predicted_{type_model}.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')

    merge_QOL = pd.merge(qol_df, qol_mod[['date', 'qualityoflife']], how="outer", on=["date"])
    merge_QOL.to_csv(f'{create_model}predicted_qual_{type_model}.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')



def linear_regression(X_train, X_test, y_train, y_test, myfile, values, daily, rolling_avg_temp_col, daylight_hours_col):
    """
    Make a linear regression model and calculate RMSE and R2

    """
    # ---Linear Regression-----------------------------------------
    line_reg = LinearRegression()
    line_reg.fit(X_train, y_train)
    print('Intercept: ', line_reg.intercept_)
    print('Coefficients: ', line_reg.coef_)
    # Predicting the y_value of linear regression
    lin_reg_y_predicted = line_reg.predict(X_test)

    # calculating the rmse
    lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_y_predicted))
    print('lin_reg_rmse : ', lin_reg_rmse)

    # Linear Regression Accuracy with test set
    lin_reg_acc = r2_score(y_test, lin_reg_y_predicted)
    print('Linear Regression R2: ', lin_reg_acc)

    myfile.writelines(f'linear regression\t{values}\t{len(values)}\t{daily}\t{rolling_avg_temp_col}\t{daylight_hours_col}\t{lin_reg_rmse}\t{lin_reg_acc}\n')
    return line_reg, myfile
    


def polynomial_linear_regression(X_train, X_test, y_train, y_test, myfile, values, daily, rolling_avg_temp_col, daylight_hours_col):
    """
    Make a polynomial linear regression model and calculate RMSE and R2
    
    """
    # ---Polynomial Linear regression-----------------------------
    poly_reg = PolynomialFeatures(degree=2)
    X_poly_train, X_poly_test = poly_reg.fit_transform(X_train), poly_reg.fit_transform(X_test)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_poly_train, y_train)
    print('Intercept: ', poly_reg_model.intercept_)
    print('Coefficients: ', poly_reg_model.coef_)

    # Predicting the y_value of polynomial linear regression
    poly_reg_y_predicted = poly_reg_model.predict(X_poly_test)

    # calculating the rmse
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    print('Polynomial_Regression_reg_rmse : ', poly_reg_rmse)

    # Plynomial Regression Accuracy with test set
    poly_reg_acc = r2_score(y_test, poly_reg_y_predicted)
    print('Plynomial Regression R2: ', poly_reg_acc)
    
    myfile.writelines(f'Plynomial Regression\t{values}\t{len(values)}\t{daily}\t{rolling_avg_temp_col}\t{daylight_hours_col}\t{poly_reg_rmse}\t{poly_reg_acc}\n')
    return poly_reg, poly_reg_model, myfile


def random_forest_regression(X_train, X_test, y_train, y_test, myfile, values, daily, rolling_avg_temp_col, daylight_hours_col):
    """
    Make a random forest regression model and calculate RMSE and R2
    
    """
    # ---Random forest regression------------------------------------
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_reg.fit(X_train, y_train)

    # Predicting the y_value of random forest
    rf_reg_y_predicted = rf_reg.predict(X_test)

    # calculating the rmse
    rf_reg_rmse = np.sqrt(mean_squared_error(y_test, rf_reg_y_predicted))
    print('random_forest_reg_rmse : ', rf_reg_rmse)

    # Random Forest Regression Accuracy with test set
    rf_reg_acc = r2_score(y_test, rf_reg_y_predicted)
    print('Random Forest Regression R2: ', rf_reg_acc)
    myfile.writelines(f'Random Forest Regression\t{values}\t{len(values)}\t{daily}\t{rolling_avg_temp_col}\t{daylight_hours_col}\t{rf_reg_rmse}\t{rf_reg_acc}\n')
    return rf_reg, myfile



def run_models(qol_df, total_df, rolling_avg_temp_col, daylight_hours_col, myfile, create_model):
    """
    Make different models
    """
    # Select columns for model
    qol_mod = qol_df[['date', 'qualityoflife', rolling_avg_temp_col, 'new_deaths', 'stringency_index',
                      daylight_hours_col, 'size_responsedate']] 
    # Save file
    qol_mod.to_csv(f'{create_model}for_models.tsv.gz', sep='\t', encoding='utf-8' ,
                           compression='gzip')
    
    
    value_list = [[rolling_avg_temp_col, 'new_deaths', 'stringency_index', daylight_hours_col]]
    for values in value_list:
        qol_mod = qol_df[['date', 'qualityoflife'] + values + ['size_responsedate']]
        qol_mod.drop_duplicates(inplace=True)
        qol_mod = qol_mod.dropna()
        # Make y and x for model
        y = qol_mod['qualityoflife'].values  # Target Variable
        X = qol_mod[values].values  # Feature Matrix
        # Standardize features by removing the mean and scaling to unit variance.
        X = StandardScaler().fit_transform(X)
        # Filter on date
        total_filter = total_df[(total_df['date'] > '2019-12-31') & (total_df['date'] < '2023-01-01')]
        total_filter = total_filter[['date'] + values]
        total_filter.drop_duplicates(inplace=True)
        total_filter = total_filter.dropna()
        X_total = total_filter[values].values.reshape(-1, len(values)) #Feature Matrix
        # Standardize features by removing the mean and scaling to unit variance.
        X_total = StandardScaler().fit_transform(X_total)

        # Split arrays or matrices into random train and test subsets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('linear regression')
        line_reg, myfile = linear_regression(X_train, X_test, y_train, y_test, myfile, values, 'new_deaths', rolling_avg_temp_col, daylight_hours_col)
        predict_values(qol_mod, total_filter, line_reg, '', X, X_total, f'linear_regression', create_model)

        # print('polynomial linear regression')
        # poly_reg, poly_reg_model, myfile = polynomial_linear_regression(X_train, X_test, y_train, y_test, myfile, values, 'new_deaths', rolling_avg_temp_col, daylight_hours_col)
        # predict_values(qol_mod, total_filter, poly_reg_model, poly_reg, X, X_total, f'polynomial_linear_regression', create_model)

        # print('random forest regression')
        # rf_reg, myfile = random_forest_regression(X_train, X_test, y_train, y_test, myfile, values, 'new_deaths', rolling_avg_temp_col, daylight_hours_col)
        # predict_values(qol_mod, total_filter, rf_reg, '', X, X_total, f'random_forest_regression', create_model)
    return myfile

    



def main():
    config = get_config()
    path_read_QOL = config['path_read_QOL']
    create_model = config['create_model']
    # Add different data to one dataframe
    final_dataframe, total_df, df_participants = add_weather_QOL(path_read_QOL, create_model) 
    final_dataframe, total_df, df_participants = add_hospitalization(final_dataframe, total_df, df_participants, path_read_QOL)
    final_dataframe, total_df, df_participants = add_stringency_index(final_dataframe, total_df, df_participants, path_read_QOL)
    final_dataframe, total_df, df_participants = sunrise_sunset(final_dataframe, total_df, df_participants, path_read_QOL)
    df_corr = add_other_cat(final_dataframe, path_read_QOL)
    
    # Calculate correlation between QOL and different variables (Table ...)
    calculate_cor(df_corr, create_model)    

    # # Write files
    # total_df_nan.to_csv(f'{create_model}4_values_per_date.tsv.gz', 
    #                     sep='\t', encoding='utf-8', compression='gzip')

    # final_dataframe.to_csv(f'{create_model}merge_final_dataframe_with_rollingavg.tsv.gz', sep='\t', encoding='utf-8' ,
    #                        compression='gzip')


    myfile = open(f'{create_model}overview_predicted.tsv', 'w')
    myfile.writelines(f'model\tvalues\tlen(values)\tdaily_hos\trolling_temp\tdaylight\tRMSE\tR2\n')
    # Call run_models
    myfile = run_models(final_dataframe, total_df, '7_max_temp', '7day_daylight_hours', myfile, create_model)
    myfile.close()

    print('DONE')


if __name__ == '__main__':
    main()