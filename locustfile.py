"""
Load testing for Fraud Detection API
Run with: locust -f locustfile.py --host http://YOUR_EXTERNAL_IP
"""

from locust import HttpUser, task, between
import random

class FraudDetectionUser(HttpUser):
    """Simulates users making fraud detection requests"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    @task(5)
    def predict_fraud(self):
        """Main prediction task (higher weight)"""
        # Generate random transaction features
        transaction = {
            "Time": random.uniform(0, 172800),
            "V1": random.uniform(-3, 3),
            "V2": random.uniform(-3, 3),
            "V3": random.uniform(-3, 3),
            "V4": random.uniform(-3, 3),
            "V5": random.uniform(-3, 3),
            "V6": random.uniform(-3, 3),
            "V7": random.uniform(-3, 3),
            "V8": random.uniform(-3, 3),
            "V9": random.uniform(-3, 3),
            "V10": random.uniform(-3, 3),
            "V11": random.uniform(-3, 3),
            "V12": random.uniform(-3, 3),
            "V13": random.uniform(-3, 3),
            "V14": random.uniform(-3, 3),
            "V15": random.uniform(-3, 3),
            "V16": random.uniform(-3, 3),
            "V17": random.uniform(-3, 3),
            "V18": random.uniform(-3, 3),
            "V19": random.uniform(-3, 3),
            "V20": random.uniform(-3, 3),
            "V21": random.uniform(-3, 3),
            "V22": random.uniform(-3, 3),
            "V23": random.uniform(-3, 3),
            "V24": random.uniform(-3, 3),
            "V25": random.uniform(-3, 3),
            "V26": random.uniform(-3, 3),
            "V27": random.uniform(-3, 3),
            "V28": random.uniform(-3, 3),
            "Amount": random.uniform(0, 500)
        }
        
        self.client.post("/predict", json=transaction, name="/predict")
    
    @task(1)
    def check_health(self):
        """Health check task"""
        self.client.get("/health", name="/health")
    
    @task(1)
    def check_ready(self):
        """Readiness check task"""
        self.client.get("/ready", name="/ready")
