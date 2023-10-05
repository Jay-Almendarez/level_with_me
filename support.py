import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.model_selection import train_test_split
from modeling import dt_comp, knn_comp, rf_comp, model_comp
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report

def wrangle():
    '''
    wrangle will read the existing csv and begin cleaning it by correcting columns, filling and dropping nulls, restricting data to be more globally representative and mapping categorical features into numerical data for exploration and statistical analysis
    '''
    df = pd.read_csv('Video_Games.csv')
    df.columns = df.columns.str.lower()
    df = df.drop(columns={'critic_score', 'critic_count', 'user_score', 'user_count','developer', 'rating'})
    null_val = df[df['year_of_release'].isnull()]
    for i in null_val.index:
        if null_val.name[i][-4:].isnumeric():
            df['year_of_release'][i] = int(null_val.name[i][-4:]) - 1
    df = df.dropna()
    df['year_of_release'] = df['year_of_release'].astype(int)
    df = df.drop(columns = 'name')
    df = df[(df.year_of_release >1993) & (df.year_of_release < 2017)]
    df = df[(df.global_sales - df.na_sales != 0) & (df.global_sales - df.eu_sales != 0) & (df.global_sales - df.jp_sales != 0) & (df.global_sales - df.other_sales != 0)]
    df = df[df.global_sales > 0.09]
    df['platform'] = df['platform'].map({'PS2': 'c', 'DS': 'h', 'PS':'c', 'X360':'c', 'PS3':'c', 'Wii':'c',
                        'XB':'c', 'GBA':'h', 'PC':'c', 'PSP':'h', 'GC':'c', 'PS4':'c',
                        'N64':'r', '3DS':'h', 'XOne':'c', 'PSV':'h', 'WiiU':'h', 'GB':'h',
                        'SNES':'r', 'DC':'r', 'GEN':'r', 'SAT':'r', 'SCD':'r' })
    df['platform'] = df['platform'].map({'c':'home_console', 'h':'handheld', 'r':'retro'})
    df = df.drop(columns = ['publisher', 'other_sales'])
    df.genre = df.genre.map({'Action':0, 'Sports':1, 'Misc':2, 'Role-Playing':3, 'Shooter':4,
                             'Adventure':5, 'Racing':6, 'Platform':7, 'Simulation':8,
                             'Fighting':9, 'Strategy':10, 'Puzzle':11}).astype(int)
    df.platform= df.platform.map({'retro':0, 'home_console':1, 'handheld':2})
    return df

def explore(df):
    '''
    explore will take our wrangles data and plot the primary finding in our exploration phase
    '''
    df = df
    plt.figure(figsize=(30,24))
    x='genre'
    plt.subplot(2,2,2)
    y='na_sales'
    sns.barplot(x=x, y=y, data=df)
    plt.title(f'{x} compared to {y}')
    plt.subplot(2,2,3)
    y='eu_sales'
    sns.barplot(x=x, y=y, data=df)
    plt.title(f'{x} compared to {y}')
    plt.subplot(2,2,4)
    y='jp_sales'
    sns.barplot(x=x, y=y, data=df)
    plt.title(f'{x} compared to {y}')
    plt.subplot(2,2,1)
    y='global_sales'
    sns.barplot(x=x, y=y, data=df)
    plt.title(f'{x} compared to {y}')
    return plt.show()

df = wrangle()

train_validate, test = train_test_split(df, test_size=0.2, random_state=117, stratify=df['genre'])
train, validate = train_test_split(train_validate, test_size=0.3, random_state=117, stratify=train_validate['genre'])

train['baseline'] = train['genre'].value_counts().idxmax()
baseline_accuracy = (train.baseline == train.genre).mean()

X_train_g = train.drop(columns=['genre', 'baseline'])
y_train_g = train['genre']

X_validate_g = validate.drop(columns=['genre'])
y_validate_g = validate['genre']

X_test_g = test.drop(columns=['genre'])
y_test_g = test['genre']

X_train_na = train.drop(columns=['genre', 'global_sales', 'eu_sales', 'jp_sales', 'baseline'])
y_train_na = train['genre']

X_validate_na = validate.drop(columns=['genre', 'global_sales', 'eu_sales', 'jp_sales'])
y_validate_na = validate['genre']

X_test_na = test.drop(columns=['genre', 'global_sales', 'eu_sales', 'jp_sales'])
y_test_na = test['genre']

X_train_eu = train.drop(columns=['genre', 'global_sales', 'na_sales', 'jp_sales', 'baseline'])
y_train_eu = train['genre']

X_validate_eu = validate.drop(columns=['genre', 'global_sales', 'na_sales', 'jp_sales'])
y_validate_eu = validate['genre']

X_test_eu = test.drop(columns=['genre', 'global_sales', 'na_sales', 'jp_sales'])
y_test_eu = test['genre']

X_train_jp = train.drop(columns=['genre', 'global_sales', 'eu_sales', 'na_sales', 'baseline'])
y_train_jp = train['genre']

X_validate_jp = validate.drop(columns=['genre', 'global_sales', 'eu_sales', 'na_sales'])
y_validate_jp = validate['genre']

X_test_jp = test.drop(columns=['genre', 'global_sales', 'eu_sales', 'na_sales'])
y_test_jp = test['genre']


# fitting train set to our optimal hyperparameters
def opt_global():
    '''
    opt_global will take the x and y split of the global data and return the optimal model found after tuning hyperparameters
    '''
    rf = RandomForestClassifier(max_depth=6)
    rf.fit(X_train_g, y_train_g)
    train_score = rf.score(X_train_g, y_train_g)
    validate_score = rf.score(X_validate_g, y_validate_g)
    test_score = rf.score(X_test_g, y_test_g)
    return print(f'Baseline Accuracy: \n{baseline_accuracy:2%}\n\
Train: \n{train_score:2%}\n\
Validate: \n{validate_score:2%}\n\
Test: \n{test_score:2%}')


# fitting train set to our optimal hyperparameters
def opt_na():
    '''
    opt_na will take the x and y split of the individual regional data for North America and return the optimal model found after tuning hyperparameters
    '''
    rf = RandomForestClassifier(max_depth = 6)
    rf.fit(X_train_na, y_train_na)
    train_score = rf.score(X_train_na, y_train_na)
    validate_score = rf.score(X_validate_na, y_validate_na)
    test_score = rf.score(X_test_na, y_test_na)
    return print(f'Baseline Accuracy: \n{baseline_accuracy:2%}\n\
Train: \n{train_score:2%}\n\
Validate: \n{validate_score:2%}\n\
Test: \n{test_score:2%}')


# fitting train set to our optimal hyperparameters
def opt_eu():
    '''
    opt_eu will take the x and y split of the individual regional data for Europe and return the optimal model found after tuning hyperparameters
    '''
    rf = RandomForestClassifier(max_depth = 5)
    rf.fit(X_train_eu, y_train_eu)
    train_score = rf.score(X_train_eu, y_train_eu)
    validate_score = rf.score(X_validate_eu, y_validate_eu)
    test_score = rf.score(X_test_eu, y_test_eu)
    return print(f'Baseline Accuracy: \n{baseline_accuracy:2%}\n\
Train: \n{train_score:2%}\n\
Validate: \n{validate_score:2%}\n\
Test: \n{test_score:2%}')


# fitting train set to our optimal hyperparameters
def opt_jp():
    '''
    opt_jp will take the x and y split of the individual regional data for Japan and return the optimal model found after tuning hyperparameters
    '''
    dt = DecisionTreeClassifier(max_depth = 4)
    dt.fit(X_train_jp, y_train_jp)
    train_score = dt.score(X_train_jp, y_train_jp)
    validate_score = dt.score(X_validate_jp, y_validate_jp)
    test_score = dt.score(X_test_jp, y_test_jp)
    return print(f'Baseline Accuracy: \n{baseline_accuracy:2%}\n\
Train: \n{train_score:2%}\n\
Validate: \n{validate_score:2%}\n\
Test: \n{test_score:2%}')