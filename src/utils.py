
#! Here we are writting the code , function class which can be used in anywhere throughtout project.

import sys
import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score


#* Add the root folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.model_selection import GridSearchCV
from src.exception import CustomeException



#* this function is used to store the object in the give file path
def save_object(file_path,obj):
    try:
        
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    
    except Exception as e:
        
        raise CustomeException(e,sys)





##** this function take the dict of model's ie. instances as input and  return the report of the model, with trained model

def evaluate_model(X_train,y_train,X_test,y_test,models:dict,param: dict):
    
    try:
        
        report=dict()
        
        for model_name,model in models.items():
            
            ##* finding the best parameter to the model :
            # if model_name not in param:
            #     raise CustomeException(f'Mode name :{model_name} is not preset in {param}',sys)
            
            gc=GridSearchCV(model,param[model_name],cv=5)
            
            gc.fit(X_train,y_train)
            
            model.set_params(**gc.best_params_)
            # model.set_params(**gs.best_params_)
            model.fit(X_train,y_train) ## * Training the model
            
            y_train_pred=model.predict(X_train)
            
            y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(y_train,y_train_pred)
            
            test_model_score=r2_score(y_test,y_test_pred)
            
            report[model_name] = {
                "test_score": test_model_score,
                "train_score": train_model_score,
                "model": model
            }
            
        return report
    
    except Exception as e:
        
        raise CustomeException(e,sys)





def load_object(file_path):
    
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        
        raise CustomeException(e,sys)




