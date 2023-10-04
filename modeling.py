# initial imports for functions and otherwise
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt

def dt_comp(X_train, y_train, X_validate, y_validate):
    '''
    dt_comp will determine in a range of 1 to 30, what the best max depth for a decision tree model is by graphing the x and y split set side by side for visual comparison.
    '''
    k_range = range(1,30)
    X_train = X_train
    y_train = y_train
    X_validate = X_validate
    y_validate = y_validate
    train_score = []
    validate_score = []
    for k in k_range:
        clf = DecisionTreeClassifier(max_depth=k, random_state=117)
        clf.fit(X_train, y_train)
        train_score.append(clf.score(X_train, y_train))
        validate_score.append(clf.score(X_validate, y_validate))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def rf_comp(X_train, y_train, X_validate, y_validate):
    '''
    rf_comp will determine in a range of 1 to 30, what the best max depth for a random forest model is by graphing the x and y split side by side for visual comparison.
    '''
    k_range = range(1,30)
    X_train = X_train
    y_train = y_train
    X_validate = X_validate
    y_validate = y_validate
    train_score = []
    validate_score = []
    for k in k_range:
        rf = RandomForestClassifier(max_depth = k, random_state=117)
        rf.fit(X_train, y_train)
        train_score.append(rf.score(X_train, y_train))
        validate_score.append(rf.score(X_validate, y_validate))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def knn_comp(X_train, y_train, X_validate, y_validate):
    '''
    knn_comp will determine in a range of 1 to 30, what the optimal amount of n_neighbors is for a knn model by graphing the x and y splits input side by side for visual comparison.
    '''
    k_range = range(1,30)
    X_train = X_train
    y_train = y_train
    X_validate = X_validate
    y_validate = y_validate
    train_score = []
    validate_score = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        train_score.append(knn.score(X_train, y_train))
        validate_score.append(knn.score(X_validate, y_validate))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def model_comp(X_train, y_train, X_validate, y_validate, max_depth1, max_depth2, n_neighbors):
    '''
    model_comp will take the x and y splits of our data and the optimal hyperparameters for our three best performing classification models and print the two splits to help compare the best model for use.
    '''
    X_train = X_train
    y_train = y_train
    X_validate = X_validate
    y_validate = y_validate
    # Best KNN Model
    knn1 = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn1.fit(X_train, y_train)

    knn_tr_acc = knn1.score(X_train,y_train)



    knn_val_acc = knn1.score(X_validate, y_validate)


    # Best Random Forest Model
    rf1 = RandomForestClassifier(max_depth=max_depth2)

    rf1.fit(X_train, y_train)

    rf_tr_acc = rf1.score(X_train, y_train)




    rf_val_acc = rf1.score(X_validate, y_validate)



    # Best Decision Tree Model
    clf1 = DecisionTreeClassifier(max_depth=max_depth1)

    clf1.fit(X_train, y_train)

    dt_tr_acc = clf1.score(X_train, y_train)




    dt_val_acc = clf1.score(X_validate, y_validate)
    
    
    
    # Best Logistic Regression Model
    lr1 = LogisticRegression()

    lr1.fit(X_train, y_train)

    lr_tr_acc1 = lr1.score(X_train,y_train)




    lr_val_acc1 = lr1.score(X_validate, y_validate)
    
    return print(f'Model Train and Validate Accuracy Scores: \n\n\
    Decision Tree Train Score: \n{dt_tr_acc:2%}\n\
    Decision Tree Validate Score: \n{dt_val_acc:2%}\n\n\
    Random Forest Train Score: \n{rf_tr_acc:2%}\n\
    Random Forest Validate Score: \n{rf_val_acc:2%}\n\n\
    K Nearest Neighbor Train Score: \n{knn_tr_acc:2%}\n\
    K Nearest Neighbor Validate Score: \n{knn_val_acc:2%}\n\n\
    ')