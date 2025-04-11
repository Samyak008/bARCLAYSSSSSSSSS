import json
import numpy as np
import rrcf
import time
from datetime import datetime
import matplotlib.pyplot as plt

class FileBasedAnomalyAgent:
    def __init__(self):
        self.num_trees = 100
        self.shingle_size = 4
        self.tree_size = 256
        
    def load_logs_from_file(self, filename, limit=1000):
        """Load logs from a JSON file"""
        logs = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line.strip()))
                        if len(logs) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading logs: {e}")
        
        print(f"Loaded {len(logs)} logs from {filename}")
        return logs
    
    def detect_anomalies(self, logs, metric_field="response_time", threshold=0.8):
        """
        Detects anomalies in logs using RRCF algorithm
        """
        if not logs:
            print("No logs to analyze")
            return [], [], []
            
        # Extract data points
        data_points = []
        contexts = []
        
        for log in logs:
            if metric_field in log:
                data_points.append(float(log[metric_field]))
                contexts.append(log)
        
        if not data_points:
            print(f"No {metric_field} values found in data")
            return [], [], []
            
        print(f"Analyzing {len(data_points)} data points")
            
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
        
        # Normalize scores for better interpretability
        min_score = min(avg_codisp) if len(avg_codisp) > 0 else 0
        max_score = max(avg_codisp) if len(avg_codisp) > 0 else 1
        score_range = max_score - min_score
        
        # Avoid division by zero
        if score_range > 0:
            normalized_scores = [(score - min_score) / score_range for score in avg_codisp]
        else:
            normalized_scores = avg_codisp
        
        # Find anomalies
        anomalies = []
        anomaly_indices = []
        anomaly_scores = []
        
        for i, score in enumerate(normalized_scores):
            if score > threshold:
                anomalies.append({
                    "timestamp": contexts[i].get('@timestamp'),
                    "score": float(score),
                    "raw_score": float(avg_codisp[i]),  # Store the original score too
                    "value": data_points[i],
                    "context": contexts[i],
                    "is_anomaly": True
                })
                anomaly_indices.append(i)
                anomaly_scores.append(score)
        
        print(f"Analysis complete. Found {len(anomalies)} anomalies.")
        
        # Plot the results
        self.plot_anomalies(data_points, anomaly_indices, anomaly_scores)
        
        return anomalies, anomaly_indices, anomaly_scores
    
    def plot_anomalies(self, data_points, anomaly_indices, anomaly_scores):
        """Plot the data points and highlight anomalies"""
        plt.figure(figsize=(12, 6))
        plt.plot(data_points, label='Response Time')
        
        if anomaly_indices:
            plt.scatter(anomaly_indices, [data_points[i] for i in anomaly_indices], 
                        color='red', label='Anomalies')
            
            # Annotate the top 5 anomalies with their scores
            top_anomalies = sorted(zip(anomaly_indices, anomaly_scores), 
                                  key=lambda x: x[1], reverse=True)[:5]
            
            for idx, score in top_anomalies:
                plt.annotate(f"Score: {score:.2f}", 
                             xy=(idx, data_points[idx]),
                             xytext=(10, 10),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.title('API Response Times with Anomaly Detection')
        plt.xlabel('Data Point Index')
        plt.ylabel('Response Time (ms)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('anomaly_detection_results.png')
        print(f"Plot saved to anomaly_detection_results.png")

def main():
    """Test the file-based anomaly detection"""
    agent = FileBasedAnomalyAgent()
    
    # Load logs from file
    logs = agent.load_logs_from_file('api_logs.json')
    
    if logs:
        # Detect anomalies
        anomalies, _, _ = agent.detect_anomalies(logs, threshold=0.9)
        
        # Print a few anomalies
        print("\nSample anomalies:")
        for i, anomaly in enumerate(anomalies[:3]):
            print(f"Anomaly {i+1}:")
            print(f"  Score: {anomaly['score']:.2f}")
            print(f"  Value: {anomaly['value']:.2f} ms")
            print(f"  API: {anomaly['context'].get('api_name')}")
            print(f"  Environment: {anomaly['context'].get('environment')}")
            print()

if __name__ == "__main__":
    main()