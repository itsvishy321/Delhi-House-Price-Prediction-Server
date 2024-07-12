import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,send_from_directory,render_template
from flask_cors import CORS
import pickle
# from dotenv import load_dotenv
# load_dotenv()

app = Flask(__name__, static_folder='../client/dist/static',
            template_folder='../client/dist')
cors = CORS(app , origins='*')

data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('Model.pkl','rb'))



@app.route("/api/localities" , methods=['GET'])
def localities():
    locality = sorted(data['Locality'].unique())
    return jsonify(
        {
            'locality' : locality
        }
    )

@app.route("/api/predict" , methods=['POST'])
def predict():
    features = request.get_json()
    bhk =  int(features['bhk'])
    furnishing = str(features['furnishing'])
    Housetype = str(features['type'])
    transaction = str(features['transaction'])
    bathroom = float(features['bathroom'])
    locality = str(features['locality'])
    area = float(features['area'])
    psqft = float(features['psqft'])

    test_data = {
    'BHK': bhk,
    'Furnishing': furnishing,
    'Type': Housetype,
    'Transaction': transaction,
    'Bathroom': bathroom,
    'Locality': locality,
    'Area': area,
    'Per_Sqft': psqft
    }


    input_df = pd.DataFrame([test_data])
    
    prediction = pipe.predict(input_df)[0]
    return jsonify({'prediction':np.round(prediction,2) })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0' , port=5000)