# Rina Rast
# Astronomy 9506S
# February 22, 2023

# This code utilizes data from AAVSO VSX (https://www.aavso.org/vsx/)
# to classify RRab stars based on their magnitude and period, using machine
# learning techniques. 

# import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
import sklearn.metrics

plt.rcParams["font.family"] = "Times New Roman"

def remove_rows(df, column, targets):
    """
    Remove rows where a specific letter or symbol is contained
    in a given column.
    
    df : str
        name of dataframe to be edited.
    column : string
        column of the dataframe to be edited.
    target : list
        letters or symbols to be removed from the dataframe.
    Returns
    df : pandas dataframe
        edited dataframe after rows have been removed.

    """
    for i in range(len(targets)):
        df = df[df[column].str.contains(targets[i]) == False]
    return df


def calculate_magrange(df):
    """
    Calculate the magnitude range from the magnitude column 
    of a given dataframe

    df : pandas dataframe
        dataframe containing magnitude column to be used.
    Returns
    df : pandas dataframe
        dataframe with additional column containing magnitude ranges as floats.

    """
    df['Mag_Range'] = ""
    mag_split = df['Mag'].str.split('-', expand=True)
    mag_range = round((mag_split[1].astype(float)-mag_split[0].astype(float)), 4)
    df['Mag_Range'] = mag_range
    return df


def svm_analysis(kernel, train_data, train_labels, test_data, test_labels, C, gamma, degree):
    """
    Perform analysis using support vector machines.

    kernel : str
        name of kernel to be used (e.g. linear, rbf, poly, etc).
    train_data : pandas dataframe
        period and magnitude values to be used in training.
    train_labels : pandas dataframe
        binary identifiers to be used in training (1=RRab, 0=other).
    test_data : pandas dataframe
        period and magnitude values to be used in testing.
    test_labels : pandas dataframe
        binary identifiers to be used in testing (1=RRab, 0=other).
    C : int
        controls smoothness of decision boundary.
    gamma : int
        defines how much influence a single training example has.
    degree : int
        degree of the polynomial to be fit to the data.
        
    Returns: classification report

    """
    svc = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    svc.fit(train_data, train_labels)
    predicted_labels = svc.predict(test_data)
    cm = confusion_matrix(test_labels, predicted_labels, labels=svc.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
    disp.plot()
    disp.ax_.set_title('SVM with '+ kernel + ' kernel')
    plt.show()
    print('SVM with', kernel, 'kernel:\n', classification_report(test_labels, predicted_labels))
    return classification_report(test_labels, predicted_labels)


def regressor_analysis(regressor, train_data, train_labels, test_data, test_labels):
    """
    Perform analysis using regression.
    
    regressor : sklearn ensemble function
        name of the regressor to be used.
    train_data : pandas dataframe
        period and magnitude values to be used in training.
    train_labels : pandas dataframe
        binary identifiers to be used in training (1=RRab, 0=other).
    test_data : pandas dataframe
        period and magnitude values to be used in testing.
    test_labels : pandas dataframe
        binary identifiers to be used in testing (1=RRab, 0=other).
        
    Returns: classification report

    """
    reg = regressor
    reg.fit(train_data, train_labels)
    predicted_labels = reg.predict(test_data)
    cm = confusion_matrix(test_labels, predicted_labels.round())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_title(regressor)
    plt.show()
    print(regressor, '\n', classification_report(test_labels, predicted_labels.round()))
    return classification_report(test_labels, predicted_labels.round())


def neural_network(classifier, train_data, train_labels, test_data, test_labels):
    """
    Perform analysis using a neural network classifier.

    classifier : sklearn neural network function
        neural network classifier to be used.
    train_data : pandas dataframe
        period and magnitude values to be used in training.
    train_labels : pandas dataframe
        binary identifiers to be used in training (1=RRab, 0=other).
    test_data : pandas dataframe
        period and magnitude values to be used in testing.
    test_labels : pandas dataframe
        binary identifiers to be used in testing (1=RRab, 0=other).
        
    Returns: classification report

    """
    classifier.fit(train_periodmag, train_labels)
    print('Training set score:', classifier.score(train_periodmag, train_labels))
    print('Test set score:', classifier.score(test_periodmag, test_labels))
    y_pred = mlp.predict(test_periodmag)
    cm = sklearn.metrics.confusion_matrix(test_labels, y_pred, labels=classifier.classes_)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    disp.ax_.set_title('{:.3} Classifier'.format(str(classifier)))
    plt.show()
    print(classifier, '\n', sklearn.metrics.classification_report(test_labels, y_pred))
    return sklearn.metrics.classification_report(test_labels, y_pred)
    
# read in the csv containing RRabs from VSX with mag < 16
rrab_data = pd.read_csv('vsx_rrab.csv', usecols=['Coords', 'Type', 'Period', 'Mag'])
 
# Data from VSX is not immediately useable in our analysis, so first we will
# clean it up a little 

    
# restrict dataset to rows where magnitude is listed as a range in V
rrab_data = rrab_data[rrab_data['Mag'].str.contains(' V') == True]
rrab_data = rrab_data[rrab_data['Mag'].str.contains('-') == True]

# remove rows where period is not listed or is uncertain
rrab_data = remove_rows(rrab_data, 'Period', ['--', ':', '<', '>'])

# remove rows where the magnitude is not listed or is uncertain   
rrab_data = remove_rows(rrab_data, 'Mag', ['--', ':', '<', '>', 'c', '  '])

# remove V from the mag column so we can use mag values as floats
rrab_data['Mag'] = rrab_data['Mag'].str.replace('V', '')
rrab_data['Mag'] = rrab_data['Mag'].str.replace(' ', '')

# calculate the magnitude ranges, rounded to 4 decimal places and saved in the last column
rrab_data = calculate_magrange(rrab_data)

####################################################

# load the file containing ~1 million variable stars with mag < 16
allvars_data = pd.read_csv('vsx_allvars.csv', usecols=['Coords', 'Type', 'Period', 'Mag'])

# restrict dataset to rows where magnitude is listed as a range in V
allvars_data = allvars_data[allvars_data['Mag'].str.contains(' V') == True]
allvars_data = allvars_data[allvars_data['Mag'].str.contains('-') == True]

# remove uncertain mag values and values given in both V and other filters
allvars_data = remove_rows(allvars_data, 'Mag', [':', '--', '<', '>', 'c', 'C', 'p', 'T', 'G', 'g', 'B', 'R', '  '])

# remove rows where period is not listed or is uncertain
allvars_data = remove_rows(allvars_data, 'Period', [':', '--', '<', '>'])

# remove rows where any type of RR Lyrae is included
allvars_data = remove_rows(allvars_data, 'Type', ['RR'])

# remove V from the mag column so we can use mag values as floats
allvars_data['Mag'] = allvars_data['Mag'].str.replace('V', '')
allvars_data['Mag'] = allvars_data['Mag'].str.replace(' ', '')

# calculate the magnitude ranges, rounded to 4 decimal places and saved in the last column
allvars_data = calculate_magrange(allvars_data)

# randomly select a subsample of the random variable data that is the same length 
# as the RRab data
allvars_data = allvars_data.sample(n=len(rrab_data))

##################################################
# create scatter plot contrasting RRabs and all other variables 

# plot the RR Lyrae data period vs. mag
plt.scatter(rrab_data['Period'].astype(float), rrab_data['Mag_Range'].astype(float), s=4, c='firebrick', label = 'RRab', alpha=0.5)

# plot the random variable stars data period vs. mag
plt.scatter(allvars_data['Period'].astype(float), allvars_data['Mag_Range'].astype(float), s=4, c='black', label='Other', alpha=0.5)

plt.title('RRabs vs. Assorted Variable Stars', fontsize=20)
plt.xlabel('Period (days)', fontsize=15)
plt.ylabel('Magnitude Range (V)', fontsize=15)

plt.xlim(0, 2)
plt.ylim(0, 2)
plt.legend(loc='upper right', fontsize=12)

#################################################

# create a combined dataset with RRabs marked as 1 and all others marked as 0
rrabs = pd.DataFrame(
	{'p': rrab_data['Period'], 
	 'm': rrab_data['Mag_Range'], 
	 'class': np.ones(len(rrab_data))})
allvars = pd.DataFrame(
	{'p': allvars_data['Period'], 
	 'm': allvars_data['Mag_Range'], 
	 'class': np.zeros(len(allvars_data))})

# combine the RRab dataset with the dataset of random variables 
combined_variables = pd.concat([rrabs, allvars])

# shuffle the data
combined_variables = combined_variables.sample(frac=1)

# test-train split (80% training, 20% testing)
train_periodmag, test_periodmag = train_test_split(combined_variables, test_size=0.2)

# extract the training data
train_data = train_periodmag[['p', 'm']]
train_labels = train_periodmag['class']

# extract the test data
test_data = test_periodmag[['p', 'm']]
test_labels = test_periodmag['class']

#################################################
# SVM classifiers:
    
# try a linear SVM classifier
svm_linear = svm_analysis('linear',  train_data, train_labels, test_data, test_labels, C=2, gamma=20, degree=0)


# try a polynomial SVM classifier 
svm_poly = svm_analysis('poly',  train_data, train_labels, test_data, test_labels, C=2, gamma=20, degree=1)


# try a radial basis kernel SVM classifier 
svm_rbf = svm_analysis('rbf',  train_data, train_labels, test_data, test_labels, C=2, gamma=20, degree=0)


# try a sigmoid SVM classifier 
svm_sigmoid = svm_analysis('sigmoid',  train_data, train_labels, test_data, test_labels, C=2, gamma=20, degree=0)

#################################################
# Regressors:

# try random forest regressor
random_forest = regressor_analysis(RandomForestRegressor(), train_data, train_labels, test_data, test_labels)


# try gradient boosting regressor
random_forest = regressor_analysis(GradientBoostingRegressor(), train_data, train_labels, test_data, test_labels)

#################################################
# Neural network classifiers:

# initialize the classifier
mlp = MLPClassifier(hidden_layer_sizes=(50, ), solver='adam', verbose=True, 
                    random_state=1, learning_rate='adaptive', 
                    learning_rate_init=0.002, max_iter=50, 
                    early_stopping=True, validation_fraction=0.2,
                    n_iter_no_change=20
                    )

# try a multi-layer perceptron classifier
mlp_classifier = neural_network(mlp, train_data, train_labels, test_data, test_labels)