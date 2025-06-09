
#* Importing the Required module's
import os 
import sys


#* Add the root folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformatioConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


#* Creating the class for Storing the path of the training ,testing ,raw data together

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    
    
    
#* Creating the Another Class for Getting the Data from server,database's api, and storing them into the above location


class DataIngestion:
    def __init__(self):
        
        self.ingestion_config=DataIngestionConfig()
        
        
    
    def initiate_data_ingestion(self):
        
        logging.info("Enter the data ingestion method or component")
        
        try:
            #& Here the code of Data Ingestion from server,database's,files,api's
            
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            logging.info("Train test Split Initiated.")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            
                      
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data intigestion is Completed.')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            
            raise CustomeException(e,sys)
            
            





if __name__ =='__main__' :
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_tranformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print("Accuracy Score of Best Model :",model_trainer.initiate_model_trainer(train_arr,test_arr))
    


































