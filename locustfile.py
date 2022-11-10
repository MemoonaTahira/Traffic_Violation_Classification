from locust import task
from locust import between
from locust import HttpUser

sample = {
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
class MLZoomUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:

            locust -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)