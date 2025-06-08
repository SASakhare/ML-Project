
#! Here we are writting the code , function class which can be used in anywhere throughtout project.

import sys
import os
import numpy as np
import pandas as pd
import dill


#* Add the root folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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









