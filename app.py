from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import lightgbm as ltb
from dependency import department_dic,region_dic,education_dic_na
app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        KPIs_met = int(request.form['KPIs_met>80%'])
        awards = int(request.form['awards_won?'])
        department = department_dic[request.form['department']]
        region = region_dic[request.form['region']]
        education = education_dic_na[request.form['education_NA']]
        training_no = int(request.form['no_of_trainings'])
        service = int(request.form['length_of_service'])
        training_score = int(request.form['avg_training_score'])
        rating = float(request.form['previous_year_rating_na'])
        
        prediction=model.predict([[KPIs_met,awards,department,training_score,rating,region,training_no,education,gender,service]])
        output=prediction[0]
        if output==0:
            return render_template('index.html',prediction_text="As per our Prediction, You would be not Promoted.")
        else:
            return render_template('index.html',prediction_text="Congratulations!! You will be promoted.")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)