import numpy as np
import pandas as pd
import bentoml
from bentoml.io import JSON
from typing import Dict, Any
# helper library to cast ndarray into numpy array, to receive e.g. a CSV file that's sent line by line
# from bentoml.io import NumpyNdarray
from pydantic import BaseModel
#import sklearn

model_name_and_tag = "traffic_violation_classification:wvsmj6dbeol4qaav"
# pydantic class for data validation
class TrafficViolationApp(BaseModel):
    subagency: str
    belts: int
    personal_injury: int
    property_damage: int
    fatal: int
    commercial_license: int
    hazmat: int
    commercial_vehicle: int
    alcohol: int
    work_zone: int
    state: str
    vehicletype: str
    make: str
    model: str
    color: str
    charge: str
    article: str
    contributed_to_accident: int
    race: str
    gender: str
    driver_city: str
    driver_state: str
    dl_state: str
    arrest_type: str
    season_of_stop: str
    hour_of_stop: str
    car_age: float


# To get model by tag:
# model_ref = bentoml.xgboost.get("credit_risk_model:g6c6ytcqm22agaav")
# To get model by latest:
model_ref = bentoml.xgboost.get(model_name_and_tag)
model_pipeline = model_ref.custom_objects['model_pipeline']

# model runner is bentoml's abstraction of the model itself, helps us access the model
model_runner = model_ref.to_runner()
svc = bentoml.Service("traffic_violation_classifier", runners = [model_runner])


# the decorater allows us to use rest and curl APIs
@svc.api (input =JSON(pydantic_model= TrafficViolationApp), output=JSON())

# def classify(application_data):
# adding async parallelizes at the endpoint level
# async def classify(traffic_violation):
async def classify(input_series: TrafficViolationApp) -> Dict[str, any]:
    # credit_application is an object of class CreditApplication
    # application data is a dict object, and incase we used @svc.api (input =JSON(), output=JSON()),
    # we could pass it to classify and use it without converting
    # application_data = traffic_violation.dict() 

    input_df = pd.DataFrame([input_series.dict()])
    # vector = model_pipeline.transform(application_data)
    vector = model_pipeline.transform(input_df)


    # prediction = model_runner.predict.run(vector)
    # also async at inference level, the runner abstraction here is doing microbatching, i.e. like a test set
    # instead of performing inference for each single user/test sample
    # corresponding change in train.ipynb by adding customizations besides the custom objects

    # results = iris_clf_runner.predict.run(input_df).to_list()
    prediction = await model_runner.predict.async_run(vector)
    print("the prediction is:", prediction)
    result = prediction[0]
    if result==0:
        return {"status": "Citation"}
    elif result == 1:
        return{"status":"Warning"}
    elif result == 2: 
        return{"status":"ERO"}


    return {"status": "Unknown"}


# P.S. Always return the type defined in svc.api output
