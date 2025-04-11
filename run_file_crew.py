from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
from file_based_crew import FileCrew
import os

def test_file_anomaly_detection():
    """Test the file-based anomaly detection independently"""
    print("=" * 80)
    print("Starting File-Based Anomaly Detection Test")
    print("=" * 80)
    
    # Create agent
    agent = FileBasedAnomalyAgent()
    
    # Load logs from file
    logs = agent.load_logs_from_file('api_logs.json')
    
    if logs:
        # Detect anomalies
        anomalies, indices, scores = agent.detect_anomalies(logs, threshold=0.8)
        
        # Print results
        print(f"\nFound {len(anomalies)} anomalies from {len(logs)} logs")
        
        # Show top anomalies
        if anomalies:
            print("\nTop anomalies:")
            # Sort anomalies by score for display
            sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
            for i, anomaly in enumerate(sorted_anomalies[:5]):
                print(f"Anomaly {i+1}:")
                print(f"  Score: {anomaly['score']:.2f}")
                print(f"  Value: {anomaly['value']:.2f} ms")
                print(f"  API: {anomaly['context'].get('api_name')}")
                print(f"  Environment: {anomaly['context'].get('environment')}")
                print()
        else:
            print("No anomalies detected with current threshold.")
    else:
        print("No logs available for analysis")
    
    return anomalies if logs and anomalies else None

def test_file_crew(anomalies=None):
    """Test the file-based CrewAI integration"""
    print("\n" + "=" * 80)
    print("Starting CrewAI Analysis")
    print("=" * 80)
    
    # Create the FileCrew
    crew = FileCrew()
    
    # Run a monitoring cycle
    result = crew.run_monitoring_cycle('api_logs.json')
    
    print("\nCrewAI Analysis Complete!")
    return result

def check_logs_exist():
    """Check if logs exist or generate them"""
    if not os.path.exists('api_logs.json') or os.path.getsize('api_logs.json') == 0:
        print("No log file found. Please run simulate_logs/simuate_logs.py first to generate logs.")
        return False
    return True

def main():
    print("Starting File-Based Anomaly Detection and CrewAI Test Suite")
    
    # Check if logs exist
    if not check_logs_exist():
        return
        
    # Run anomaly detection
    anomalies = test_file_anomaly_detection()
    
    # Run CrewAI analysis
    test_file_crew(anomalies)

if __name__ == "__main__":
    main()