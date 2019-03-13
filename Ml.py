import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

#Read dataset.csv
dataset = pd.read_csv("bank-additional-full.csv",sep = ';')

#store y values in res
res = dataset.iloc[:,-1]
res = res.replace(['yes','no'],(1,0))
#print(dataset.head())


#Label encoding so that coloumns having string values can be trained.
dataset_train = dataset
f = ['job','marital','default','education','housing','loan','contact','month','poutcome','day_of_week']
for i in f:
    __ = preprocessing.LabelEncoder()
    __ = __.fit(dataset_train[i])
    dataset_train[i] = __.transform(dataset_train[i])

features = dataset_train.drop(['y'],axis = 1)

#Feature Selection by using Feature importance.
X = features  
y = res
etc = ExtraTreesClassifier()
etc.fit(X,y)
fimp = pd.Series(etc.feature_importances_, index=X.columns)
fimp.nlargest(10).plot(kind='barh')
plt.show()
#Feature Selection by using correlation matrix.
print(dataset.corr())

#Preparing data.
rfeatures = features[['duration','euribor3m','age','campaign','job','education','day_of_week','marital']]
xTrain, xTest, yTrain, yTest = train_test_split(rfeatures, res, test_size = 0.2, random_state = 42 )


#Random Forest
rf = RandomForestClassifier(n_estimators = 150)
rf.fit(xTrain,yTrain)
acc_test = rf.score(xTest,yTest)
print("Accuracy for Random Forest :"+str(acc_test))

#KNN
from  sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(xTrain,yTrain)
yPredict = knc.predict(xTest)
print("Accuracy for KNN :" + str(accuracy_score(yPredict,yTest)))

#NeuralNet
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(alpha = .88,hidden_layer_sizes = 30)
nn.fit(xTrain,yTrain)
yPredict = nn.predict(xTest)
print("Accuracy for Neural Nets :"+str(accuracy_score(yPredict,yTest)))
