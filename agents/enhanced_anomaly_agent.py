import json
import numpy as np
import rrcf
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

class EnhancedAnomalyAgent:
    def __init__(self):
        self.num_trees = 100  # Increased from default
        self.shingle_size = 4
        self.tree_size = 256
        self.min_samples = 30  # Minimum samples needed for detection
        
    def load_logs_from_file(self, filename, limit=5000):  # Increased limit for better learning
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
    
    def preprocess_data(self, logs, metric_field="response_time"):
        """Preprocess logs to extract features and normalize"""
        if not logs:
            return None, None, None
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([{
            'timestamp': log.get('@timestamp'),
            'response_time': log.get(metric_field, 0),
            'api_name': log.get('api_name', 'unknown'),
            'environment': log.get('environment', 'unknown'),
            'status_code': log.get('status_code', 200),
            'correlation_id': log.get('correlation_id', '')
        } for log in logs])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add features
        df['is_error'] = (df['status_code'] >= 400).astype(int)
        
        # One-hot encode categorical variables
        api_dummies = pd.get_dummies(df['api_name'], prefix='api')
        env_dummies = pd.get_dummies(df['environment'], prefix='env')
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # Add sequential features
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['response_time', 'time_delta', 'hour', 'minute']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Combine all features
        feature_df = pd.concat([df[numerical_cols], api_dummies, env_dummies], axis=1)
        
        return df, feature_df, logs
    
    def detect_anomalies(self, logs, metric_field="response_time", threshold=0.8, 
                        use_multidimensional=True, contextual_analysis=True):
        """
        Enhanced anomaly detection using RRCF with optional multidimensional features
        """
        if not logs or len(logs) < self.min_samples:
            print("Insufficient logs for anomaly detection")
            return [], [], []
            
        print(f"Analyzing {len(logs)} data points")
        
        # Preprocess data
        original_df, feature_df, _ = self.preprocess_data(logs, metric_field)
        
        if original_df is None or feature_df is None:
            return [], [], []
            
        # Select data for RRCF
        if use_multidimensional and feature_df.shape[1] > 1:
            # Use multiple features for detection (more accurate but slower)
            data_points = feature_df.values
            print(f"Using {data_points.shape[1]} features for anomaly detection")
        else:
            # Use just response time (faster but less accurate)
            data_points = np.array([[x] for x in original_df['response_time'].values])
            print("Using only response time for anomaly detection")
            
        # Process with RRCF
        forest = []
        n_samples = len(data_points)
        
        # Sampling strategy
        sample_size_ratio = 0.7  # Use 70% of points per tree
        sample_size = min(n_samples, self.tree_size, int(n_samples * sample_size_ratio))
        
        print(f"Building forest with {self.num_trees} trees, each with {sample_size} points")
        
        for _ in range(self.num_trees):
            # Create a sample of indices
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            
            # Create a tree from the sample
            tree = rrcf.RCTree()
            
            # Add points to the tree
            for index in indices:
                point = data_points[index]
                tree.insert_point(point, index=index)
            
            # Add tree to the forest
            forest.append(tree)
        
        # Compute anomaly scores
        avg_codisp = np.zeros(n_samples)
        for tree in forest:
            for i in tree.leaves:
                avg_codisp[i] += tree.codisp(i) / self.num_trees
        
        # Normalize scores
        min_score = min(avg_codisp) if len(avg_codisp) > 0 else 0
        max_score = max(avg_codisp) if len(avg_codisp) > 0 else 1
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = [(score - min_score) / score_range for score in avg_codisp]
        else:
            normalized_scores = avg_codisp
            
        # Apply contextual thresholds if enabled
        if contextual_analysis:
            # Group by API and environment to apply contextual thresholds
            original_df['anomaly_score'] = normalized_scores
            
            # Different thresholds for different APIs based on their normal variance
            api_thresholds = {}
            for api in original_df['api_name'].unique():
                api_scores = original_df[original_df['api_name'] == api]['anomaly_score']
                if len(api_scores) > 10:  # Need enough samples
                    # Higher variance APIs need higher thresholds
                    api_var = original_df[original_df['api_name'] == api]['response_time'].var()
                    base_threshold = threshold
                    # Adjust threshold based on variance, capped between 0.7 and 0.95
                    api_thresholds[api] = min(max(base_threshold - 0.1 + (api_var * 0.15), 0.7), 0.95)
                else:
                    api_thresholds[api] = threshold
            
            print("API-specific thresholds:")
            for api, th in api_thresholds.items():
                print(f"  {api}: {th:.2f}")
            
            # Find anomalies with contextual thresholds
            anomaly_indices = []
            for i, score in enumerate(normalized_scores):
                api = original_df.iloc[i]['api_name']
                api_threshold = api_thresholds.get(api, threshold)
                if score > api_threshold:
                    anomaly_indices.append(i)
        else:
            # Simple threshold
            anomaly_indices = [i for i, score in enumerate(normalized_scores) if score > threshold]
        
        # Create anomaly objects
        anomalies = []
        anomaly_scores = []
        
        for i in anomaly_indices:
            # Get context from the original dataframe
            context = original_df.iloc[i].to_dict()
            timestamp = context.pop('timestamp', None)
            
            if timestamp:
                timestamp = timestamp.isoformat()
            
            anomalies.append({
                "timestamp": timestamp,
                "score": float(normalized_scores[i]),
                "raw_score": float(avg_codisp[i]),
                "value": float(original_df.iloc[i]['response_time']),
                "context": logs[i],  # Original log entry for full context
                "is_anomaly": True
            })
            anomaly_scores.append(normalized_scores[i])
        
        print(f"Analysis complete. Found {len(anomalies)} anomalies ({len(anomalies)/len(logs)*100:.1f}%).")
        
        # Plot the results
        self.plot_enhanced_anomalies(original_df, anomaly_indices, normalized_scores)
        
        return anomalies, anomaly_indices, anomaly_scores
    
    def plot_enhanced_anomalies(self, df, anomaly_indices, scores):
        """Plot enhanced visualization with time series and multiple dimensions"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Response time with anomalies over time
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['response_time'], 'b-', alpha=0.5, label='Response Time')
        
        if anomaly_indices:
            anomaly_times = df.iloc[anomaly_indices]['timestamp']
            anomaly_values = df.iloc[anomaly_indices]['response_time']
            plt.scatter(anomaly_times, anomaly_values, color='red', s=50, label='Anomalies')
            
        plt.title('API Response Times with Anomalies')
        plt.ylabel('Normalized Response Time')
        plt.xlabel('Time')
        plt.legend()
        
        # Format x-axis to show readable timestamps
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        
        # Plot 2: Scores distribution
        plt.subplot(2, 1, 2)
        plt.hist(scores, bins=30, alpha=0.7, color='skyblue', label='All Scores')
        if anomaly_indices:
            plt.hist([scores[i] for i in anomaly_indices], bins=10, 
                     alpha=0.7, color='red', label='Anomaly Scores')
        plt.axvline(x=0.8, color='r', linestyle='--', label='Threshold')
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('advanced_anomaly_detection_results.png')
        print(f"Enhanced plot saved to advanced_anomaly_detection_results.png")

def main():
    """Test the enhanced anomaly detection"""
    agent = EnhancedAnomalyAgent()
    
    # Load logs from file
    logs = agent.load_logs_from_file('api_logs.json')
    
    if logs:
        # Detect anomalies with enhanced algorithm
        anomalies, _, _ = agent.detect_anomalies(
            logs, 
            threshold=0.8, 
            use_multidimensional=True,
            contextual_analysis=True
        )
        
        # Print a few anomalies
        print("\nSample anomalies:")
        for i, anomaly in enumerate(sorted(anomalies, key=lambda x: x['score'], reverse=True)[:5]):
            print(f"Anomaly {i+1}:")
            print(f"  Score: {anomaly['score']:.2f}")
            print(f"  Value: {anomaly['value']:.2f} ms")
            print(f"  API: {anomaly['context'].get('api_name')}")
            print(f"  Environment: {anomaly['context'].get('environment')}")
            print(f"  Status Code: {anomaly['context'].get('status_code', 'unknown')}")
            print()

if __name__ == "__main__":
    main()