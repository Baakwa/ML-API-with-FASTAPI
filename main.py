# #
# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# #Get an app for creating API endpoint
# app = FastAPI()

# #Create a class
# class PatientDetails(BaseModel):
#     #Name each feature and data structure
#     Plasma_glucose : int
#     Blood_Work_R1 : int
#     Blood_Pressure : int
#     Blood_Work_R2 : int
#     Blood_Work_R3 : int
#     BMI : float
#     Blood_Work_R4 : float
#     Patient_age : int
#     Insurance : int

# #Load all machine learning models and label encoder
# Logistic_pipeline = joblib.load('./Models/LogisticR_model.joblib')
# Gradient_pipeline = joblib.load('./Models/GradientM_model.joblib')
# RandomF_pipeline = joblib.load('./Models/RandomF_model.joblib')
# encoder = joblib.load('./Models/encoder.joblib')


# #Add a decorator for home page 
# @app.get('/')

# #Create an instance of FastAPI 
# def home_page():
#     return {'message': 'Welcome to A Sepsis Prediction API!'}

#Define data to be used using pydantic

# #Add a decorator For Logistic regression Model
# @app.post('/predict_Logistic_Regression')

# #Write an endpoint for each model
# def Logistic_regression_predict(data:patient_details):

# #Convert model to a dictionary and then a dataframe 
#     #prediction_df = pd.DataFrame([data.model_dump()])
#     prediction_df = pd.DataFrame([data.model_dump()])
    
#     #Make prediction for  model
#     LR_prediction = Logistic_pipeline.predict(prediction_df)

#     #Convert predictions to an int 
#     #prediction_df = int(prediction_df)
    

#     #Select prediction value
#     LR_prediction = LR_prediction[0]


#     #Inverse transform prediction values
#     LR_prediction_lab = encoder.inverse_transform([LR_prediction])[0]

#     #Get predictions probability
#     LR_prediction_prob = Logistic_pipeline.predict_proba(prediction_df)

#     return {'prediction' : LR_prediction,
#             'probability' : LR_prediction_prob}


# #Add a decorator For Random Forest Model
# @app.post('/predict_Random_Forest')

# #Write an endpoint for each model
# def Random_forest_predict(data:patient_details):

# #Convert model to a dictionary and then a dataframe 
#     #RF_prediction_df = pd.DataFrame([data.model_dump()])
#     prediction_df = pd.DataFrame([data.model_dump()])

# #Convert predictions to an int 
#     #RF_prediction_df = int(RF_prediction_df)
    
#     #Make prediction for  model
#     RF_prediction = RandomF_pipeline.predict(prediction_df)

#     #Select prediction value
#     RF_prediction = RF_prediction[0]


#     #Inverse transform prediction values
#     RF_prediction_lab = encoder.inverse_transform([RF_prediction])[0]

#     #Get predictions probability
#     RF_prediction_prob = RandomF_pipeline.predict_proba(prediction_df)

#     return {'prediction' : RF_prediction_lab,
#             'probability' : RF_prediction_prob}


# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
# #@app.post('/predict_Gradient_Boosting')
# #@app.post('/predict_Random_Forest')

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from fastapi.encoders import jsonable_encoder

#Get an app for creating API endpoint
app = FastAPI()

# Define your machine learning models and encoder
Logistic_pipeline = joblib.load('./Models/LogisticR_model.joblib')
RandomF_pipeline = joblib.load('./Models/RandomF_model.joblib')
Gradient_pipeline = joblib.load('./Models/GradientM_model.joblib')
encoder = joblib.load('./Models/encoder.joblib')

class PatientDetails(BaseModel):
    Plasma_glucose: int
    Blood_Work_R1: int
    Blood_Pressure: int
    Blood_Work_R2: int
    Blood_Work_R3: int
    BMI: float
    Blood_Work_R4: float
    Patient_age: int
    Insurance: int

#Add a decorator for home page 
@app.get('/')

#Create an instance of FastAPI 
def home_page():
    return {'message': 'Welcome to A Sepsis Prediction API!'}

#Add a decorator For Logistic regression Model
@app.post('/predict_Logistic_Regression')

# Create Endpoint for Logistic Regression Model
def logistic_regression_predict(data: PatientDetails):
    # Convert model data to a dictionary
    data_dict = jsonable_encoder(data)
    prediction_df = pd.DataFrame([data_dict])

    # Make prediction for the Logistic Regression model
    LR_prediction = Logistic_pipeline.predict(prediction_df)

    #Make prediction probabilities
    LR_prediction_prob = Logistic_pipeline.predict_proba(prediction_df)

    # Convert prediction to a standard Python boolean
    LR_prediction = bool(LR_prediction[0])

    return {'prediction': LR_prediction,
            'probability': LR_prediction_prob[0][1]
            }


#Add a decorator For Random Forest Model
@app.post('/predict_Random_Forest')

# Endpoint for Random Forest Model
def random_forest_predict(data: PatientDetails):
    # Convert model data to a dictionary
    data_dict = jsonable_encoder(data)
    prediction_df = pd.DataFrame([data_dict])

    # Make prediction for the Random Forest model
    RF_prediction = RandomF_pipeline.predict(prediction_df)

    #Make prediction probabilities
    RF_prediction_prob = RandomF_pipeline.predict_proba(prediction_df)

    # Convert prediction to a standard Python boolean
    RF_prediction = bool(RF_prediction[0])

    return {'prediction': RF_prediction,
            'probability': RF_prediction_prob[0][1]
            }


#Add a decorator For Random Forest Model
@app.post('/predict_Gradient_Boosting')

# Endpoint for Gradient Boosting Model
def gradient_boosting_predict(data: PatientDetails):
    # Convert model data to a dictionary
    data_dict = jsonable_encoder(data)
    prediction_df = pd.DataFrame([data_dict])
    
    # Make prediction for the Gradient Boosting model
    GB_prediction = Gradient_pipeline.predict(prediction_df)
    
    #Make prediction probabilities
    GB_prediction_prob = Gradient_pipeline.predict_proba(prediction_df)
    
    # Convert prediction to a standard Python boolean  
    GB_prediction = bool(GB_prediction[0])
    
    return {'prediction': GB_prediction,
            'probability': GB_prediction_prob[0][1]
            }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
