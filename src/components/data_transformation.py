## The main purpose of the data transformation is basically to do feature engineering plus data cleaning.

import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

"""
Why use @dataclass?
To reduce boilerplate code when creating classes meant to store data (like simple objects or records).

Without @dataclass:
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age})"

p = Person("Alice", 30)
print(p)

With @dataclass:
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 30)
print(p)

"""
    
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        
        """the reason I'm creating this function okay.
This is just to create all my pickle files, uh, which will basically be responsible in doing converting
your categorical features into numerical or, sorry, categorical into numerical.
If you want to probably perform standard scalar and all and all.     

        """
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("")
            
            logging.info("Numerical columns standard scaling completed")
            
            logging.info("Categorical columns encoding completed")
            
            ## Now combine num pipeline with column pipeline with the help of column transformer
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining Preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            ## np.c_ is a handy indexing trick that lets you concatenate arrays column-wise — like putting arrays side by side to form a 2D array.
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object")
            
            ## preprocessing_obj needs to be converted in pickle file
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        
        except Exception as e:
            raise CustomException(e,sys)