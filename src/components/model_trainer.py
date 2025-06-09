
#! This Script file using for Writing the code for Model Training :

import os
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor


#* Add the root folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')




##* Creating the class for model training :

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    
    def initiate_model_trainer(self,train_array,test_array):
        
        '''
        This funtion find the best model according to the accuracy score and return it :
        '''
        
        try:
            
            logging.info('Split training and test input data')
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRFRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            
            ## * Calling the function for model trainning and getting the best model.
            
            model_report=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ## * model_report : model : accuracy_score  
            ##* to find best model accuracy_score should be high so that we sort it and find the model having accuracy score is max
            
            best_model_name = max(model_report, key=lambda name: model_report[name]["test_score"])
            best_model_info = model_report[best_model_name]

            best_model_score = best_model_info["test_score"]
            best_model = best_model_info["model"]
                        
            # print(model_report)
            if best_model_score<0.6:
                raise CustomeException('No Best Model Found.')
            
            logging.info('Best Model Found on both Training and Testing dataset')
            
            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            
            y_predict=best_model.predict(X_test)
            
            r2_scored=r2_score(y_test,y_predict)
            
            return r2_scored
            
        except Exception as e:
            
            raise CustomeException(e,sys)



















