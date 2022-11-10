# Week 7: Production-ready ML Models with BentoML

Project done in partial fulfilment for the certification of MLZoomcamp2022 by DataTalks.Club. Details can be found [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/cohorts/2022/projects.md#midterm-project).


# Problem:

Classify a traffic violation into 4 categories based on the level, severity and type of violation. The 4 different types of violations are:

- **Citation:** A written record of something wrong while operating a vehicle or while parked. Examples: Speeding tickets, Running a red light/stop sign, Driving under the influence (DUI), Failure to signal
- **Warning:** A warning is issued by the officer is a statement that the motorist has committed some offense, but is being spared the actual citation. Officers use their own discretion whether to issue a citation or warning.
- **ESERO:** Electronic Safety Equipment Repair Order
- **SERO:** Safety Equipment Repair Order

The data is from Maryland Capitol Police Department.The mojor challenge of the problem is working almost entirely with categorical variables and very high cardinality of features. 

# Dataset: 

The original dataset can be found [here](https://www.kaggle.com/datasets/felix4guti/traffic-violations-in-usa?select=Traffic_Violations.csv) on Kaggle.

This dataset is very untidy, so a cleaned version after doing cleaning and EDA in this [notebook](./TrafficViolation_Clean_EDA%20final.ipynb) can be found [here](https://github.com/MemoonaTahira/Traffic_Violation_Classification/releases/download/latest/cleaned_traffic_violations.csv). Dataset download links are added to all code file.


These are dataset stats after initial cleaning and feature engineering (basically everything that needs to be done before train-test split). 

<img src = "./imgs/dataset_stats.jpg">

The dimensions of the dataset are: `Dataframe dimensions: (1010737, 28)`

**The major work done on features:**

 - Removing duplicates
 - Reomving description of location in words(whose logitude/latitude are given), longitude, latitude and geolocation (which was a tuple of logitude and latitude), in favour of subagency, which basically denotes 7 zones where traffic violations were recorded. In absense of subagency, geolocation would have to be binned into zones. 
 - Removing features such as agency and accident that had only one value/category.
 - Lowercasing column names and string variable
 - Converting bool columns to 0 and 1 
 - Converting date to seasons by binning months, called season_of_stop
 - Convrting time to portion of day called hour_of_stop, e.g. late night or afternoon, using binning on hours.
 - Removing anamoly from year of car, e.g. year below 0 or beyond 2022, and converting it to car_age

**On target variable:** 

The traffic_violation is the target variable with 4 values:

```
citation    0.486215
warning     0.462300
esero       0.050579
sero        0.000906
Name: violation_type, dtype: float64
```

Since sero is not even 0.01 of the dataset, it is combined with esero, which itself it the minority class at about 5%. We still have a class imbalance but it is manageable as just as seen below:

<img src = "./imgs/class_imbalance.png" width=50%>

# Methodology:

The methodology followed is CRISP-DM, with roughly the same outline as week 1-7 of [MLZoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). 

We start with data cleaing and EDA, and then feature engineering and feature selection, while doing doing missing value imputations, scaling, one-hot encoding and dimensionality reduction.

Once we have a cleaned dataset, we train it on fine-tuned linear and tree based models and use validation set to select best model. The best model is the trained on full train data, and tested one last time on withheld dataset.

Finally, the model is containerized with BentoML and deployed as a classification service to predict the type of traffic violation.

Full workflow [here](./TrafficViolation_Clean_EDA%20final.ipynb) in this jupyter notebook.

# Reproducibility:

The whole project is reproducible, ideally via bentoML. Here are details how to recreate the project at every step:

## 1. Create conda environment:

Create a conda environement from requirement.txt file using the requirement.txt file [here](./requirements.txt):

```
conda create --name traffic_classify --file requirements.txt
```

(P.S. If you want to use a pip environment, you can also use [pip-requirement.txt](./pip_requirements.txt) to install it.)

Using the newly created conda environement, explore the notebook for cleaning, EDA feature engineering and selecting best model, which is the [TrafficViolation_Clean_EDA.ipynb](./TrafficViolation_Clean_EDA.ipynb)

## 2. Analysis and Results:

Have a look at model pipelines (Logistic Regression, Random Forest and XGBoost) and detailed results [here](./results.md).

## 3. Train the final model:

Using the same environment as before, i.e. traffic_classify:

Run the final_train.py file to train the best model with tuned parameters on the cleaned dataset. In a terminal with the traffic_classify environment activated, run this:

`python final_Train.py`


It will save these items:

    - [xgboost_traffic_violation_model.sav](./xgboost_traffic_violation_model.sav) 
    - [sklearn_pipeline.pkl](./sklearn_pipeline.pkl)
    - bentoML model with both the xgboost final model and custom pipeline object

Check that your model saved correctly by running `bentoml models list`

<!-- ## 4. (Optional) Build the BentoML model and serve it locally:

Run from terminal with conda env traffic_classify activated:

```
bentoml serve service.py:svc --reload
```
This step tests the bentoML model before converting it to a service.  -->


## 5. Build the BentoML service and serve it locally:

First, get list of models stored in the bentoml models directory

`bentoml models list`

Get name and tag of the model you want, e.g. in my case:

`"traffic_violation_classification:c3zxrptavo3q4vhb"`

Use you own model name and tag in [service.py](./service.py) in line 9, and make sure to use it with quotes included, i.e. `"traffic_violation_classification:c3zxrptavo3q4vhb"`


Build the bentoML service.

`bentoml build`

And you'll see this: 

<img src = "./imgs/bentobuild.jpg"> 

Serve the container by running this is the terminal:

`bentoml serve --production`

Now you can test your bentoML service now running a classification service locally.

## 6. Using Swagger UI once the service is runnnig locally

Go to any browser and open this link: 0.0.0.0:3000 OR localhost:3000

It will open the Swagger UI which looks like this:

<img src = "./imgs/swagger1.jpg">

Click on "try it out" and paste the sample test user given below and click execute:

    {
    "subagency": "4th_district,_wheaton",
    "belts": 0,
    "personal_injury": 0,
    "property_damage": 0,
    "fatal": 0,
    "commercial_license": 1,
    "hazmat": 0,
    "commercial_vehicle": 0,
    "alcohol": 0,
    "work_zone": 0,
    "state": "md",
    "vehicletype": "02_-_automobile",
    "make": "honda",
    "model": "civic",
    "color": "red",
    "charge": "21-904(b2)",
    "article": "transportation_article",
    "contributed_to_accident": 0,
    "race": "hispanic",
    "gender": "m",
    "driver_city": "silver_spring",
    "driver_state": "md",
    "dl_state": "md",
    "arrest_type": "a_-_marked_patrol",
    "season_of_stop": "summer",
    "hour_of_stop": "late_night",
    "car_age": 28.0
    }

<img src = "./imgs/swagger2.jpg">

You should see a result like this:

<img src = "./imgs/swagger3.jpg">

## 7. Do load testing with locust

open another tab in your browser while the `bentoml serve --production` command is running from terminal and run:

`locust -H <http://localhost:3000>`

Do load testing if you want to see if the model can handle appopriate load before deploying it as service to cloud.

## 8. Dockerize the bentoML service:

Start your docker service:
`sudo dockerd &`

Build your bentoML container:
```
cd bentoml/bentos/
tree
# copy name and tag of the service we built in step 5:
# Note: Models are located in /bentoml/models and have a different name and tag from BentoML services that are located in: 
# /bentoml/bentos

 bentoml containerize traffic_violation_classifier:ga4yxpdbbc676aav
```

It will take a moment to build the container. Once it is done, serve the container locally:

`docker run -it --rm -p 3000:3000 traffic_violation_classifier:ga4yxpdbbc676aav serve --production`

Test it using same steps as before from [here](#6-using-swagger-ui-once-the-service-is-runnnig-locally)

## 9. Pull bentoML service docker image from DockerHub and run it:

You can skip all the previous steps, and just pull the bentoML image from DockerHub and build a container and run it via docker like this:

```
docker pull MemoonaTahira/traffic_violation_classification
docker run -it --rm -p 3000:3000 traffic_violation_classifier:ga4yxpdbbc676aav serve --production
```
Test it using same steps as before from [here](#6-using-swagger-ui-once-the-service-is-runnnig-locally)

## 10. Deployment of bentoML as a service to Cloud: 

The bentoML container which can be found in `/home/user/bentoml/bentos` can be deployed to any cloud service as it is. 

<!-- Or to docker registry? -->

<!-- ## 11. Deploy the BentoML container to Mobegenius:

    - create an account on Mobegenius: https://studio.mogenius.com/user/registration
    - verfiy yuor account and select the free plan
    - Create a cloud space with a reflective name, e.g. TrafficViolation
    - Choose create from docker image from any registry
    - Connect to your docker hub registry. See note below for setting up DockerHub
    - Follow this tutorial exactly: https://www.youtube.com/watch?v=NMIi_DDVxAs&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=96
    - And you have deployed your bentoML container with a Swagger UI in AWS. You can then test if the traffic violation classification service inside the container is working properly by using the sample user data to get a prediction.
    - You should see the same prediction of 'citation' like this:

    <img src = "./imgs/aws">


## 12. Setting up DockerHub:

    - Create an account and verify it
    - Choose free plan and create a repository with a name that reflects your service, e.g. traffic_violation_classification
    - From your local terminal, with sudo dockerd running, run the following:
    `docker commit <existing-container> <hub-user>/<repo-name>[:<tag>]`
    - E.g. in my case it would look like this:

    - It will ask for the DockerHub username and password you used during signup, provide it and push image to your Repo. 
    - Done! :)
 -->
