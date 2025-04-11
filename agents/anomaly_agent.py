from crewai import Agent, Task
import numpy as np
from rrcf import RRCF  # Using Robust Random Cut Forest
import elasticsearch
from elasticsearch import Elasticsearch

class AnomalyAgent:
    def __init__(self, es_host="http://elasticsearch:9200"):
        self.es = Elasticsearch([es_host])
        self.rrcf = RRCF(num_trees=100, shingle_size=4)
        
    def detect_anomalies(self, index_pattern="logstash-*", time_window="15m", 
                         metric_field="response_time", threshold=0.9):
        """
        Detects anomalies in logs using RRCF algorithm
        """
        # Query recent logs
        query = {
            "range": {
                "@timestamp": {
                    "gte": f"now-{time_window}",
                    "lte": "now"
                }
            }
        }
        
        response = self.es.search(
            index=index_pattern,
            body={
                "query": query,
                "sort": [{"@timestamp": "asc"}],
                "_source": ["@timestamp", metric_field, "api_name", "correlation_id", "environment"]
            },
            size=1000
        )
        
        if not response['hits']['hits']:
            return []
            
        # Extract data points
        data_points = [hit['_source'][metric_field] for hit in response['hits']['hits']]
        timestamps = [hit['_source']['@timestamp'] for hit in response['hits']['hits']]
        contexts = [hit['_source'] for hit in response['hits']['hits']]
        
        # Process with RRCF
        anomaly_scores = self.rrcf.batch_process(data_points)
        
        # Find anomalies
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score > threshold:
                anomalies.append({
                    "timestamp": timestamps[i],
                    "score": float(score),
                    "value": data_points[i],
                    "context": contexts[i],
                    "is_anomaly": True
                })
                
                # Store anomaly back to ES
                self.es.index(
                    index="anomalies",
                    document={
                        "@timestamp": timestamps[i],
                        "anomaly_score": float(score),
                        "metric_value": data_points[i],
                        "api_name": contexts[i].get("api_name", "unknown"),
                        "environment": contexts[i].get("environment", "unknown"),
                        "correlation_id": contexts[i].get("correlation_id", "unknown")
                    }
                )
        
        return anomalies