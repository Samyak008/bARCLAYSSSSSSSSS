from crewai import Agent, Task
import numpy as np
import rrcf  # Import the package
from elasticsearch import Elasticsearch

class AnomalyAgent:
    def __init__(self, es_host="http://localhost:9200"):
        self.es = Elasticsearch([es_host])
        self.num_trees = 100
        self.shingle_size = 4
        self.tree_size = 256
        
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
        
        try:
            response = self.es.search(
                index=index_pattern,
                body={
                    "query": query,
                    "sort": [{"@timestamp": "asc"}],
                    "_source": ["@timestamp", metric_field, "api_name", "correlation_id", "environment"]
                },
                size=1000
            )
        except Exception as e:
            print(f"Error querying Elasticsearch: {e}")
            return []
        
        if not response['hits']['hits']:
            print("No data found in Elasticsearch")
            return []
            
        # Extract data points
        data_points = []
        timestamps = []
        contexts = []
        
        for hit in response['hits']['hits']:
            if metric_field in hit['_source']:
                data_points.append(float(hit['_source'][metric_field]))
                timestamps.append(hit['_source']['@timestamp'])
                contexts.append(hit['_source'])
        
        if not data_points:
            print(f"No {metric_field} values found in data")
            return []
            
        # Process with RRCF
        forest = []
        for _ in range(self.num_trees):
            # Create a sample of indices
            n = len(data_points)
            sample_size = min(n, self.tree_size)
            indices = np.random.choice(n, size=sample_size, replace=False)
            
            # Create a tree from the sample
            tree = rrcf.RCTree()
            
            # Add points to the tree
            for index in indices:
                point = np.array([data_points[index]])
                tree.insert_point(point, index=index)
            
            # Add tree to the forest
            forest.append(tree)
        
        # Compute anomaly scores
        avg_codisp = np.zeros(len(data_points))
        for tree in forest:
            for i in tree.leaves:
                avg_codisp[i] += tree.codisp(i) / self.num_trees
        
        # Find anomalies
        anomalies = []
        for i, score in enumerate(avg_codisp):
            if score > threshold:
                anomalies.append({
                    "timestamp": timestamps[i],
                    "score": float(score),
                    "value": data_points[i],
                    "context": contexts[i],
                    "is_anomaly": True
                })
                
                try:
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
                except Exception as e:
                    print(f"Error storing anomaly to Elasticsearch: {e}")
        
        print(f"Analysis complete. Found {len(anomalies)} anomalies.")
        return anomalies