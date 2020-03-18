# PCA example
# author: Allee, Hartin updated by sdm

import numpy as np  # needed for arrays
import pandas as pd  # data frame
import matplotlib.pyplot as plt  # modifying plot
from sklearn.model_selection import train_test_split  # splitting data
from sklearn.preprocessing import StandardScaler  # scaling data
from sklearn.linear_model import LogisticRegression  # learning algorithm
from sklearn.decomposition import PCA  # PCA package
from sklearn.metrics import accuracy_score  # grading
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from warnings import filterwarnings
from matplotlib.colors import ListedColormap  # for choosing colors


################################################################################
# Function to plot decision regions.                                           #
# Inputs:                                                                      #
#    X - feature values of each sample, e.g. coordinates on cartesian plane    #
#    y - the classification of each sample - a one-dimensional array           #
#    classifier - the machine learning classifier to use, e.g. perceptron      #
#    test_idx - typically the range of samples that were the test set          #
#               the default value is none; if present, highlight them          #
#    resolution - the resolution of the meshgrid                               #
# Output:                                                                      #
#    None                                                                      #
#                                                                              #
# NOTE: this will support up to 5 classes described by 2 features.             #
################################################################################

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # we will support up to 5 classes...
    markers = ('v', 'x', 'o', '^', 's')  # markers to use
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # colors to use
    cmap = ListedColormap(colors[:len(np.unique(y))])  # the color map

    # plot the decision surface
    # x1* will be the range +/- 1 of the first feature
    # x2* will be the range +/- 1 of the first feature
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # all rows, col 0
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # all rows, col 1

    # now create the meshgrid (see p14.py for examples)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # ravel flattens the array. The default, used here, is to flatten by taking
    # all of the first row, concanentating the second row, etc., for all rows
    # So we will predict the classification for every point in the grid
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # reshape will take the resulting predictions and put them into a matrix
    # with the same shape as the mesh
    Z = Z.reshape(xx1.shape)

    # using Z, create a contour plot so we can see the regions for each class
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())  # set x-axis ranges
    plt.ylim(xx2.min(), xx2.max())  # set y-axis ranges

    # plot all samples
    # NOTE: X[y==c1,0] returns all the column 0 values of X where the
    #       corresponding row of y equals c1. That is, only those rows of
    #       X are included that have been assigned to class c1.
    # So, for each of the unique classifications, plot them!
    # (In this case, idx and c1 are always the same, however this code
    #  will allow for non-integer classifications.)

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=c1)

    # highlight test samples with black circles
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]  # test set is at the end
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=55, label='test set')


# read the database. Since it lackets headers, put them in
df_mine = pd.read_csv('sonar_all_data_2.csv', header=None)
X = df_mine.iloc[:, :60].values  # features are in columns 0:59
y = df_mine.iloc[:, 60].values  # classes are in column 60!
# now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
stdsc = StandardScaler()  # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Determine the number of components that yields the best results on the test data.

accuracy_max = 0  # max accuracy
max_component = 0  # number of components that yields the best results on the test data
accuracies = []
component_nums = []
for pca_componet in range(1, 60):
    pca = PCA(n_components=pca_componet)
    X_train_pca = pca.fit_transform(X_train_std)  # apply to the train data
    X_test_pca = pca.transform(X_test_std)  # do the same to the test data
    # now create a MLP Classifier and train on it
    model = MLPClassifier(solver='lbfgs', alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)  # how do we do on the test data?
    accuracies.append(accuracy_score(y_test, y_pred))
    component_nums.append(pca_componet)
    if accuracy_max < accuracy_score(y_test, y_pred):
        accuracy_max = accuracy_score(y_test, y_pred)
        max_component = pca_componet



#For each number of components used, print the number of components and the accuracy achieved

data = {'Component':  component_nums,
        'Accuracy': accuracies,
        }
df = pd.DataFrame(data, columns=['Component', 'Accuracy'])
# print('\n', df)   # Vertical
df_transposed = df.T
print('\n', df_transposed)  #  Horizontal

# At the end, print the maximum accuracy along with the number of components that achieved  that accuracy.
print('max_component: ', max_component)
print('Max Accuracy: ', accuracy_max)

# Plot accuracy versus the number of components
plt.bar(component_nums, accuracies, facecolor='g', alpha=0.75)
plt.title('Accuracy versus the Number of Components')
plt.show()

pca = PCA(n_components=2)  # only keep two "best" features!
X_train_pca = pca.fit_transform(X_train_std)  # apply to the train data
X_test_pca = pca.transform(X_test_std)  # do the same to the test data
###########            MLP Classifier           ################
print('MLP Classifier: \n')
# now create a MLP Classifier and train on it
model = MLPClassifier(solver='lbfgs', alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)  # how do we do on the test data?
# now create a MLP Classifier and train on it
print('Number in test ', len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f\n' % accuracy_score(y_test, y_pred))

# now combine the train and test data and see how we do
X_comb_pca = np.vstack((X_train_pca, X_test_pca))
y_comb = np.hstack((y_train, y_test))
print('Number in combined ', len(y_comb))
y_comb_pred = model.predict(X_comb_pca)

print('Misclassified combined samples: %d' % (y_comb != y_comb_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_comb, y_comb_pred))

confustion_results = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', confustion_results)

classification_result = classification_report(y_test, y_pred)
print('Classification Report \n', classification_result)
filterwarnings('ignore')
# Now visualize the results
plot_decision_regions(X_train_pca, y_train, classifier=model)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
