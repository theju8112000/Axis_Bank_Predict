from flask import Flask,render_template,request,session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

global x_train, y_train, x_test, y_test
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload1',methods=["POST","GET"])
def uploaddata():
    if request.method=="POST":
        dataset = request.files['file']
        filename = dataset.filename
        file = "dataset//"+filename
        session['dataset'] = file

        return render_template('upload.html', msg="success")
        return render_template('upload.html')

@app.route('/view')
def viewdata():
    datafile = session.get('dataset')
    df=pd.read_csv(datafile)
    df=df.head(10)
    return render_template('view.html',data=df.to_html())

@app.route('/split')
def splitdata():
    return render_template('split.html')

@app.route('/splitdata',methods=["POST","GET"])
def splitdataset():
    global x_train,y_train,x_test,y_test
    global x,y
    if request.method == "POST":
        testsize=request.form['testsize']
        testsize = float(testsize)
        datafile = session.get('dataset')
        df = pd.read_csv(datafile)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=testsize)
        len1 = len(x_train)
        len2 = len(x_test)

        return render_template('split.html',msg="done",tr=len1,te=len2)
    return render_template('split.html')

@app.route('/trainmodel')
def models():
    return render_template('trainmodel.html')

@app.route('/modelpredict',methods=["POST","GET"])
def prediction():
    if request.method == "POST":
        value = int(request.form['model'])

    if value==1:
        model = LinearRegression()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        acc1=model.score(x_train,y_train)
        return render_template('trainmodel.html',msg="accuracy",acc=acc1,alg="LinearRegression")
    if value == 2:
        model = GaussianNB()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        acc2 = model.score(x_train,y_train)
        return render_template('trainmodel.html', msg="accuracy", acc=acc2,alg="GaussianNB")
    if value == 3:
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc3 = model.score(x_train, y_train)
        return render_template('trainmodel.html', msg="accuracy", acc=acc3, alg="DecisionTreeClassifier")
    if value == 4:
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc4 = model.score(x_train, y_train)
        return render_template('trainmodel.html', msg="accuracy", acc=acc4, alg="RandomForestClassifier")
    if value == 7:
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc7 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc7, alg="GradientBoostingClassifier")
    if value == 8:
        model = XGBClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc8 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc8, alg="XGBClassifier")


@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict1',methods =['POST','GET'])
def pred():
    s = []
    datafile = session.get('dataset')
    df = pd.read_csv(datafile)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    if request.method== 'POST':
        date= request.form['date']
        Previous_CP = request.form['pcp']
        OpenPrice = request.form['op']
        HighPrice = request.form['hp']
        LowPrice = request.form['lp']
        LastPrice = request.form['lap']
        ClosePrice = request.form['cp']
        s.extend([date,Previous_CP,OpenPrice,HighPrice,LowPrice,LastPrice,ClosePrice])
        model=LinearRegression()
        model.fit(X, y)
        y_pred = model.predict([s])
        return render_template('predict.html',msg="success",op=y_pred)
        #return f'<html><body><h1>{y_pred}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'

if __name__ == '__main__':
    app.secret_key = "hai"
    app.run(port=8080, debug=True)