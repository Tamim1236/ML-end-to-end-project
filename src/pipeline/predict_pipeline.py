import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
  def __init__(self):
    pass

  def predict(self, features):
    try:
      model_path = "artifacts/model.pkl"
      preprocessor_path = "artifacts/preprocessor.pkl"
      
      model = load_object(file_path = model_path)
      preprocessor = load_object(file_path = preprocessor_path)

      scaled_data = preprocessor.transform(features)
      prediction = model.predict(scaled_data)
      return prediction
    
    except Exception as e:
      raise CustomException(e, sys)

class CustomData:
  def __init__(
    self,
    gender: str,
    reading_score: int,
    writing_score: int,
    parental_level_of_education: str,
    race_ethnicity: str,
    lunch: str,
    test_preparation_score: str
    ):

    self.gender = gender
    self.reading_score = reading_score
    self.writing_score = writing_score
    self.parental_level_of_education = parental_level_of_education
    self.race_ethnicity = race_ethnicity
    self.lunch = lunch
    self.test_preparation_score = test_preparation_score
    
  # get input in the form of a pd dataframe
  def get_data_as_data_frame(self):
    try:
      custom_data_input_dict = {
        "gender" : [self.gender],
        "race_ethnicity" : [self.race_ethnicity],
        "parental_level_of_education" : [self.parental_level_of_education],
        "lunch" : [self.lunch],
        "test_preparation_score" : [self.test_preparation_score],
        "reading_score" : [self.reading_score],
        "writing_score" : [self.writing_score]
      }
      return pd.DataFrame(custom_data_input_dict)

    except Exception as e:
      raise CustomException(e, sys)