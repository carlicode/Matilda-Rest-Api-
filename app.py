from flask import Flask, jsonify, request
from transformers import pipeline
from Scientifics import scientifics
import pandas as pd
#Model Huggin Face
nlp = pipeline(
    'question-answering', 
model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
tokenizer=(
        'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',  
        {"use_fast": False}))

scientifics_CSV = pd.read_csv('scientifics_questions4 - scientifics_questions4.csv')

#Desplegamos Flask
app = Flask(__name__)

def model(c, q):
    a = nlp({
        'question': q,
        'context': c})
    return a

@app.route('/ping')
def ping():
    return jsonify({'message':'pong'}) #ESto es un objeto

@app.route('/scientifics')#Este funciona por defecto con el method GET
def getScientifics():
    return jsonify(scientifics)

@app.route('/scientific/<string:scientific_name>')
def getScientific(scientific_name):
    print(scientific_name)
    #Recorre la lista de cientificas y compara si el nombre coincide con em nombre del parametro
    scientificFound = [scientific for scientific in scientifics if scientific['name'] == scientific_name]
    if (len(scientificFound)>0):
        return jsonify({'scientific': scientificFound[0] })
    return jsonify({'message': 'Cientifica NOT FOUND'})

@app.route('/model', methods=['POST'])
def addScientific():
    new_answer = {
        "context" : request.json['context'],
        "question" : request.json['question']
    }
    answer = model(new_answer['context'],new_answer['question'])
    print(answer)
    return jsonify({'message': (answer)})

@app.route('/CSV')
def csvAccess():
    shape = str(scientifics_CSV.shape[0])
    print(shape)
    return jsonify({'CSV message': shape})


@app.route('/user', methods=['POST'])
def user():
    new_answer = {
        "woman" : request.json['woman'],
        "question" : request.json['question']
    }
    scientificsVal = scientifics_CSV.values
    for i in scientificsVal:
        if i[0] == new_answer['woman']:
            context = i[1]
            print(context)
            break
    answer = model(context, new_answer['question'])
    print(answer)
    return jsonify({'message': (answer)})





if __name__ == '__main__':
    app.run(debug=True, port=5000 )