# author: Ziru "Ron" Chen

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Given constant: use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

def get_list_of_university_towns():
    data = []
    state = 0
    with open('university_towns.txt') as file:
        for line in file:
            line = line.rstrip()
            if line.endswith('[edit]'):
                state = line[0:line.index('[')]
            else:
                try:
                    data.append([state, line[0:line.index(' (')]])
                except:
                    data.append([state, line])
    df = pd.DataFrame(data, columns=['State', 'RegionName'])
    return df

def get_recession_start():
    df = pd.read_excel('gdplev.xls', names=['quarter', 'GDP'], usecols=[4, 6], skiprows=213)
    df['growth'] = df['GDP'].diff()
    df = df[df['growth'] < 0]
    prev = 0
    for idx, val in df['growth'].items():
        if idx - prev == 1:
            return df['quarter'].loc[prev]
        prev = idx

def get_recession_end():
    df = pd.read_excel('gdplev.xls', names=['quarter', 'GDP'], usecols=[4, 6], skiprows=253)
    df['growth'] = df['GDP'].diff()
    df = df[df['growth'] > 0]
    prev = 0
    for idx, val in df['growth'].items():
        if idx - prev == 1:
            return df['quarter'].loc[idx]
        prev = idx
				
def get_recession_bottom():
    df = pd.read_excel('gdplev.xls', names=['quarter', 'GDP'], usecols=[4, 6], skiprows=253)
    df = df.loc[0:5].set_index('quarter')
    return df['GDP'].argmin()

def convert_housing_data_to_quarters():
    df = pd.read_csv('City_Zhvi_AllHomes.csv').set_index(['State', 'RegionName'])
    df = df.rename(states)
    df = df.loc[:, '2000-01':]
    df.columns = pd.to_datetime(df.columns)
    df = df.resample('Q',axis=1).mean()
    df = df.rename(columns=lambda x: str(x.to_period('Q')).lower())
    return df
	
def run_ttest():
    start = get_recession_start()
    before_start = start[:-1] + str(int(start[-1]) - 1)
    bottom = get_recession_bottom()
    utowns = get_list_of_university_towns()

    housing = convert_housing_data_to_quarters()[[before_start, bottom]]
    housing = housing.sort_index()
    housing['Price Ratio'] = housing[before_start] / housing[bottom]
    
    utown_housing = housing.merge(utowns, left_index=True, right_on=['State', 'RegionName']).set_index(['State', 'RegionName'])
    other_housing = housing.drop(utown_housing.index)
    
    statistic, pvalue = ttest_ind(utown_housing['Price Ratio'], other_housing['Price Ratio'], nan_policy = 'omit')
    diff = pvalue < 0.01
    
    better = utown_housing['Price Ratio'].mean() < other_housing['Price Ratio'].mean()
    better = 'university town' if better else 'non-university town'
    
    return (diff, pvalue, better)
