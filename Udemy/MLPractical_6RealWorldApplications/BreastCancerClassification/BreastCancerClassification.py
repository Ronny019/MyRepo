# CASE STUDY: BREAST CANCER CLASSIFICATION
# Dr. Ryan Ahmed

# STEP #1: PROBLEM STATEMENT


#- Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
#- 30 features are used, examples:
#        - radius (mean of distances from center to points on the perimeter)
#        - texture (standard deviation of gray-scale values)
#        - perimeter
#        - area
#        - smoothness (local variation in radius lengths)
#        - compactness (perimeter^2 / area - 1.0)
#        - concavity (severity of concave portions of the contour)
#        - concave points (number of concave portions of the contour)
#        - symmetry 
#        - fractal dimension ("coastline approximation" - 1)

#- Datasets are linearly separable using all 30 input features
#- Number of Instances: 569
#- Class Distribution: 212 Malignant, 357 Benign
#- Target class:
#         - Malignant
#         - Benign


#https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

#![image.png](attachment:image.png)

# STEP #2: IMPORTING DATA

# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
#%matplotlib inline

# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer)

cancer.keys()

print(cancer['DESCR'])

print(cancer['target_names'])

print(cancer['target'])

print(cancer['feature_names'])

print(cancer['data'])


cancer['data'].shape

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


df_cancer.head()

df_cancer.tail()

x = np.array([1,2,3])
x.shape

Example = np.c_[np.array([1,2,3]), np.array([4,5,6])]
Example.shape

# STEP #3: VISUALIZING THE DATA

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer['target'], label = "Count") 

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


#sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer_all, fit_reg=False)

# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 

# STEP #4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)


# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)


print(X)

y = df_cancer['target']
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)

# STEP #5: EVALUATING THE MODEL

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_predict))

# STEP #6: IMPROVING THE MODEL

min_train = X_train.min()
print(min_train)

range_train = (X_train - min_train).max()
print(range_train)

X_train_scaled = (X_train - min_train)/range_train

print(X_train_scaled)

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_predict))

# IMPROVING THE MODEL - PART 2

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train_scaled,y_train)

print(grid.best_params_)

print(grid.best_estimator_)

grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions)
print(cm)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test,grid_predictions))

# Excellent Job!
