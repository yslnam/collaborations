import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import glob

def test(a):
    return a

def file_compressor(startdir, 
                    returndir = '../', 
                    parse_dates = ['starttime', 'stoptime'], 
                    extension = 'csv', 
                    rdm = 0, 
                    pathchange = True, 
                    percent = 1, 
                    export = False, 
                    name ='combined_csv.csv'):
    '''
    Takes a folder full of files and downsizes into a single dataframe by taking a random sample.
    This function was bult using the function pd_read_downsample from HelperFn.py
    Requires importing random, os, pandas, glob.
    
    startdir = path directory to your desired files (such as csv)
    df  = name of your product dataframe
    
    returndir will return your path directory to your orginal location.
    extension is set to csv as default
    Random (rdm) seed is set to 0 as default
    pathchange controls the startpwd and returnpwd. Set this to false if yoour directory is already in list of files
    Default downsizing is set to 1% of the data
    If you want a csv copy set export to True.
    --------------------------------------------------------------------------------------------
    
    Example:
    file_compressor(startdir = 'dataFlder/csvFlder', returndir ='../..')
    '''
    if pathchange == True:
        os.chdir( startdir )
    random.seed(rdm)
    extension = extension
    df_list = []
    percent = percent/100
    
    # Uses glob fn from glob to match the extension pattern (csv). Written in regex
    all_filenames = [i for i in glob.glob('*.{}'.format(extension) )] 
    
    i = 1
    for file in all_filenames:
        try:
            n = sum(1 for line in open(file)) - 1 #number of records in file (excludes header)
            s = round(n*percent)
            skip = sorted( random.sample(range(1,n+1), n-s)) #the 0-indexed header will not be included in the skip list
            df_downsample = pd.read_csv(file,skiprows=skip)
            df_downsample.columns = df_downsample.columns.str.lower()
            df_downsample.columns = df_downsample.columns.str.replace(' ','')
            df_list.append(df_downsample)
            print('csv {} completed:{}'.format(i, file))
        except ValueError:
            print('csv read error:{}'.format(file) )
        i += 1
    
    finished_df = pd.concat(df_list)
    print('Dataframe variable is set as finished_df')
    
    if pathchange == True:
        os.chdir( returndir )
    if export == True:
        finished_df.to_csv( name, index=False, encoding='utf-8-sig') #encoding is for non-english languages
        print('finished_df printed as {}'.format(name) )
    print('Complete!')
    

def pd_read_downsample(filename, per, parse_dates = ['starttime', 'stoptime'] ):
    '''
    Takes in a filename within the same directory as the ipython file and a decimal \
    and returns a downsampled dataframe with the percentage of rows random sampled set by per
    '''
    n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
    s = round(n*per)
    skip = sorted( random.sample(range(1,n+1), n-s)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(filename, parse_dates= parse_dates, skiprows=skip)
    return df


def highest_NA(df):
    '''
    Visualizes NAs by column
    '''
    #select all the na values, sum them, sort descending
    nans = df.isna().sum().sort_values(ascending=False)
    #filter them for columns w/ na
    nans = nans[nans > 0]
    #create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid()
    ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")
    ax.set_ylabel("No. of missing values", labelpad=10)
    ax.set_xlim(-0.6, len(nans) - 0.4)
    ax.xaxis.set_tick_params(rotation=90)
    plt.show()

def colpercent(df):
    '''
    helper function to print columns with missing and the percentage missingness:
    '''
    print("Total NaN in Dataframe: " , df.isnull().sum().sum())
    print("Percent Missingness in Dataframe: ", 100*df.isnull().sum().sum()/(len(df.index)*len(df.columns)))
    print('-'*55)
    percentnulldf = df.isnull().sum()/(df.isnull().sum()+df.notna().sum())
    print("Percent Missingness by Columns:")
    print(100*percentnulldf[percentnulldf>0].sort_values(ascending=False))

def colpercount(df):
    '''
    printout to help view levels within features with missingness
    '''
    percentnulldf = df.isnull().sum()/(df.isnull().sum()+df.notna().sum())
    percent_ordered_df=percentnulldf[percentnulldf>0].sort_values(ascending=False)
    for i in range(len(percent_ordered_df)):
        print(percent_ordered_df.index[i])
        print('-'*15)
        print(train[percent_ordered_df.index[i]].value_counts())
        print('-'*55)

def zeroper(df, value):
    '''
    helper function to print out percentage of zeroes by column
    '''
    l=[]
    columns=[]
    for i in range(len(df.columns)):
        if 0 in df[df.columns[i]].value_counts():
            if 100*df[df.columns[i]].value_counts().loc[0]/len(df[df.columns[i]])>value:
                l.append((df.columns[i], 100*df[df.columns[i]].value_counts().loc[0]/len(df[df.columns[i]])))
            else:
                pass
        else:
            pass
    print(len(l))    
    print('-'*55)
    for j in range(len(l)):
        columns.append(l[j][0])
        print('Percent of zeroes: ', l[j])
        print('-'*55)
    print(columns)
    return columns

def data_eval(df):
    '''
    helper functions to characterize missingness by row and column
    '''
    for i in range(len(df.columns)):
        print('-'*50)
        print('Column Name: ', df.columns[i])
        if (df[df.columns[i]].dtypes == 'float64' or df[df.columns[i]].dtypes == 'int64') \
        and df[df.columns[i]][df[df.columns[i]]<0].count()>0:
            print('Number of negatives: ', df[df.columns[i]][df[df.columns[i]]<0].count())
        if df[df.columns[i]][df[df.columns[i]]=='None'].count() > 0:
            print('Number of None strings: ', df[df.columns[i]][df[df.columns[i]]=='None'].count())
        if df[df.columns[i]][df[df.columns[i]]==''].count() > 0:
            print('Number of empty strings: ', df[df.columns[i]][df[df.columns[i]]==''].count())
        else:
            print('Column ' + str(i) + ' has no negatives, empty strings or Nones')

def row_na_list(df, value):
    '''
    generates list of percentage missingness by row
    '''
    l=[]
    for i in range(len(df.index)) :
        if df.iloc[i].isnull().sum() > value:
            l.append([i, df.iloc[i].isnull().sum()])
    return l

def index_retrieve(df, value, measure):
    '''
    helper function to retrieve row and column index labels for correlation matrix values for greater than value when value>0 \ 
    and less than value when value<0. Prints out the values that correspond to those indices
    '''
    poslist = list()
    # Get bool dataframe with True at positions where the given value exists and filter out on-diagonal elements
    if measure == 'spearman':
        if value>0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)>value].isna().isin([value])
        if value<0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)<value].isna().isin([value])
        else:
            pass
    elif measure == 'pearson':
        if value>0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)>value].isna().isin([value])
        elif value<0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)<value].isna().isin([value])
        else:
            pass
    # Get list of columns that contains the value
    series = result.any()
    columnNames = list(series[series == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            poslist.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    if value > 0:
        print('Number of correlations with value greater than ' + str(value) + ': ' + str(len(poslist)))
    if value < 0:
        print('Number of correlations with value less than ' + str(value) + ': ' + str(len(poslist)))
    else:
        pass
    for i in range(len(poslist)):
        print('-'*40)
        print('index labels: ', poslist[i][0], poslist[i][1])
        print('value at index: ', df.corr().loc[poslist[i]])
    return poslist