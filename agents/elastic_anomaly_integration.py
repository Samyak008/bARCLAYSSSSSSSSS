import os
import time
from datetime import datetime, timedelta
import json
import traceback
from elasticsearch_connector import ElasticsearchConnector
from file_based_anomaly_agent import FileBasedAnomalyAgent
try:
    from enhanced_anomaly_agent import EnhancedAnomalyAgent
    ENHANCED = True
except ImportError:
    ENHANCED = False
try:
    from enhanced_forecasting import EnhancedForecastingAgent
    FORECASTING = True
except ImportError:
    FORECASTING = False

class ElasticsearchAnomalyMonitor:
    """Monitor that reads logs from Elasticsearch and stores analysis results back"""
    
    def __init__(self, interval=60, use_enhanced=True, index_pattern="logs-*"):
        self.interval = interval  # Monitoring interval in seconds
        self.use_enhanced = use_enhanced and ENHANCED
        self.es_connector = ElasticsearchConnector()
        self.index_pattern = index_pattern
        
        # Initialize agents
        if self.use_enhanced:
            self.anomaly_agent = EnhancedAnomalyAgent()
            print("Using Enhanced Anomaly Agent")
        else:
            self.anomaly_agent = FileBasedAnomalyAgent()
            print("Using Standard Anomaly Agent")
            
        if FORECASTING:
            self.forecasting_agent = EnhancedForecastingAgent()
            print("Forecasting capability available")
            
        # Create index templates
        self.es_connector.create_index_template()
    
    def get_logs_from_elasticsearch(self, minutes=10):
        """Get logs from the last N minutes"""
        end_time = datetime.now().isoformat()
        start_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        logs = self.es_connector.query_time_range(
            start_time=start_time,
            end_time=end_time,
            index_pattern=self.index_pattern,
            size=1000
        )
        
        print(f"Retrieved {len(logs)} logs from Elasticsearch")
        return logs
    
    def process_logs(self, logs):
        """Process logs with anomaly detection and forecasting"""
        if not logs:
            print("No logs to process")
            return None, None
            
        # Detect anomalies
        if self.use_enhanced:
            try:
                anomalies, indices, scores = self.anomaly_agent.detect_anomalies(
                    logs, 
                    threshold=0.8, 
                    use_multidimensional=True,
                    contextual_analysis=True
                )
            except Exception as e:
                print(f"Error in enhanced anomaly detection: {e}")
                traceback.print_exc()
                # Fall back to standard detection
                anomalies, indices, scores = self.anomaly_agent.detect_anomalies(logs, threshold=0.8)
        else:
            anomalies, indices, scores = self.anomaly_agent.detect_anomalies(logs, threshold=0.8)
            
        print(f"Found {len(anomalies)} anomalies in {len(logs)} logs")
        
        # Store anomalies in Elasticsearch
        for anomaly in anomalies:
            self.es_connector.store_anomaly(anomaly)
        
        # Generate forecasts if available
        forecasts = None
        if FORECASTING:
            try:
                forecasts = self.forecasting_agent.generate_forecasts(logs, anomalies)
                print(f"Generated forecasts for {len(forecasts)} APIs")
                
                # Store forecasts in Elasticsearch
                for api_name, forecast in forecasts.items():
                    forecast['api_name'] = api_name
                    self.es_connector.store_forecast(forecast)
            except Exception as e:
                print(f"Error in forecasting: {e}")
                traceback.print_exc()
        
        return anomalies, forecasts
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        print(f"\n=== Starting monitoring cycle at {datetime.now().isoformat()} ===")
        
        # Get logs from Elasticsearch
        logs = self.get_logs_from_elasticsearch(minutes=10)
        
        # Process logs
        anomalies, forecasts = self.process_logs(logs)
        
        # Store summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'logs_analyzed': len(logs) if logs else 0,
            'anomalies_found': len(anomalies) if anomalies else 0,
            'forecasts_generated': len(forecasts) if forecasts else 0,
            'anomaly_apis': list(set([a['context'].get('api_name') for a in anomalies])) if anomalies else [],
            'forecast_apis': list(forecasts.keys()) if forecasts else []
        }
        
        self.es_connector.store_document('monitoring-summaries', summary)
        print(f"Monitoring cycle complete. Found {summary['anomalies_found']} anomalies.")
        
    def run_continuous(self):
        """Run monitoring continuously at specified interval"""
        print(f"Starting continuous monitoring (interval: {self.interval}s)")
        try:
            while True:
                self.run_monitoring_cycle()
                print(f"Waiting {self.interval} seconds until next cycle...")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            
    def run_once(self):
        """Run monitoring once and exit"""
        self.run_monitoring_cycle()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run anomaly detection on Elasticsearch logs')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--standard', action='store_true', help='Use standard anomaly detection instead of enhanced')
    parser.add_argument('--index', type=str, default="logs-*", help='Index pattern to query')
    
    args = parser.parse_args()
    
    # Create and run the monitor
    monitor = ElasticsearchAnomalyMonitor(
        interval=args.interval,
        use_enhanced=not args.standard,
        index_pattern=args.index
    )
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run_continuous()