#!/usr/bin/env python3
# -*- coding: utf-8 -*-



### for BIOS534 Spring 2019 python prepration QA session
### @author: Yanting Huang <yanting.huang@emory.edu>

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
"""
"""

if __name__ == '__main__':
    ### data path specification here
    #use relative path here since the script is in the same directory as the data file
    train = "HW_1_training.txt"
    test  = "HW_1_testing.txt"


    ### load the data into pandas dataframe
    train_df = pd.read_csv(train, sep=",") #separator is always important; use "," for this small dataset
    test_df = pd.read_csv(test, sep=",")
    print(train_df.head().T) # get a peek at what each column contains
    print(train_df.shape) # know the shape of the data
    print(train_df.columns) # get the list of column names
    print(train_df.dtypes) # know the data types of each column (float, int, object(string) and etc.)

    #calculate the mean vector and covariance matrix of each class based on the training data.
    mu = np.mean(train_df, axis =1) #arguments are (array, dimension to avg on). gives column avg
    cov = np.cov(train_df, rowvar = 0) #col are var
    
    #Construct a Bayesian decision boundary using prior calculated from the data
    
    
    
    #Plot the decision boundary on the scatter plots of the training data. 
    
    
    
    #Calculate the classification error rate on the testing data. 












    ### generate univariate data summary
    df_summary = data_df.describe() #very helpful function in pandas to show you the critical summary statistics

    ### bivariate plotting
    # the following kaggle blog is good to read through
    # useful when you want to do explotary data analysis (EDA)
    # https://www.kaggle.com/residentmario/bivariate-plotting-with-pandas
    # scatter plot API: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.scatter.html
    # only works when the feature is numeric

    # you can try different figure size and different plot type beyond scatter plot
    # the following code is an example of how to draw subplot using pandas and matplotlib
    # you can draw the bi-variate plot one-by-one or only draw the plot with the features of your choice
    X = data_df.iloc[:, 1:]
    X_numeric = X.loc[:, X.dtypes!='object'] #(150, 7)
    y = data_df.iloc[:, 0]
    num_samples = X_numeric.shape[0]
    num_numeric_feats = X_numeric.shape[1]
    num_bivar_figs = 21 #we may want a plot with a 7 subplots in a row and 3 rows in total
    fig, ax = plt.subplots(nrows=3, ncols=7, figsize=(50,80))
    feat_ind_list = []
    for i in range(num_numeric_feats):
        for j in range(i+1, num_numeric_feats):
            feat_ind_list.append((i,j))
    #plot starts from here
    print(feat_ind_list)
    sub_plot_ind = 0
    for row in ax:
        for col in row:
            X_numeric.plot(x=feat_ind_list[sub_plot_ind][0], y=feat_ind_list[sub_plot_ind][1], ax=col, kind='scatter')
            sub_plot_ind += 1
    plt.savefig("bivariate_plot")

    ### fit a simple linear model for y ~ x1 to x5
    # here I used sklearn.linear_model.LinearRegression() to illustrate
    # set normalize=True to make each predictor in a same scale
    # you can set n_jobs=-1 if you want to fully use your multi cpus to accelerate training
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    selected_predictors = ['x1', 'x2', 'x3', 'x4', 'x5']
    X_sub = X[selected_predictors]
    print(X_sub['x5'].value_counts())
    # x5 is categorical data; you need to preprocess it before fitting the model
    # also note x5 is not ordinal so that we can one-hot-encoding it
    # here I dropped one column for one-hot-encoded feature x5 to reduce multicollinearity
    # this is reasonable since 2 out 3 one-hot-encoded features already contain full information
    X_sub_ohe = pd.get_dummies(X_sub, drop_first=True)
    linear_regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lr_model = linear_regressor.fit(X_sub_ohe, y)
    # get the fitted coefficient
    lr_model_coef = lr_model.coef_
    # try prediction on training set
    train_set_predicitons = lr_model.predict(X_sub_ohe)
    # training r^2
    train_r2 = lr_model.score(X_sub_ohe, y) #0.8796959722309944

