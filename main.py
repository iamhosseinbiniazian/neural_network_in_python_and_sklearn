import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

##############Load Data#########################
data=pd.read_csv('pima-indians-diabetes.csv')
data=data.values
X,Y=data[:,:-1],data[:,-1].astype('int')
############################preprocess Data###################################
X = preprocessing.scale(X)
#################Train , Test Split(Cross validation)###################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
############################################################################################


########Train Multilayer Perceptron and report all score#################
clf_mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1,activation='tanh')
clf_mlp.fit(X_train,Y_train)
predict_mlp=clf_mlp.predict(X_test)
print("************Multilayer Perceptron Report********************")
print("Accuracy Multilayer Perceptron is {:.2f} %".format(accuracy_score(Y_test,predict_mlp)*100))
print("ROC Naive Bayes is {:.2f}".format(roc_auc_score(Y_test,predict_mlp)))
print(classification_report(Y_test,predict_mlp))
print("Confusion Matrix Multilayer Perceptron is :")
print(confusion_matrix(Y_test,predict_mlp))
print("************End of Multilayer Perceptron Report********************")
##############End of Multilayer Perceptron Part#####################3