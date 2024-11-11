from flask import Flask,request,render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
dtr=pickle.load(open('dtr.pkl','rb'))
pp=pickle.load(open('pp.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=='POST':
        temperature=request.form['temperature']
        fertilizer=request.form['fertilizer']
        nitrogen=request.form['nitrogen']
        potassium=request.form['potassium']
        phosphorous=request.form['phosphorous']
        
        features=np.array([[fertilizer,temperature,nitrogen,phosphorous,potassium]],dtype=object)
        trans_features=pp.transform(features)
        prediction=dtr.predict(trans_features).reshape(-1,1)

        return render_template('index.html',prediction=prediction[0][0])
if __name__=="__main__":
    app.run(debug=True)
