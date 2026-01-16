from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

x,y = load_iris(return_X_y=True)
xtrain,xtest, ytrain,ytest = train_test_split(x,y , test_size=0.2)

for c in [0.1,1.0]:
    with mlflow.start_run():
        m = LogisticRegression(C=c, max_iter=200)
        m.fit(xtrain, ytrain)
        acc_score = accuracy_score(ytest, m.predict(xtest))
        mlflow.log_param("C",c)
        mlflow.log_metrics('acc',acc_score)
        mlflow.sklean.log_model(m, 'model')

mlflow ui
