from agents.anomaly_agent import AnomalyAgent
from agents.correlation_agent import CorrelationAgent
from agents.prediction_agent import PredictionAgent
from agents.response_agent import ResponseAgent

def test_anomaly_detection():
    """Test the anomaly detection agent independently"""
    print("Testing Anomaly Detection...")
    agent = AnomalyAgent(es_host="http://localhost:9200")
    
    # Test connection to Elasticsearch
    try:
        info = agent.es.info()
        print(f"Connected to Elasticsearch: {info.get('version', {}).get('number')}")
    except Exception as e:
        print(f"Failed to connect to Elasticsearch: {e}")
        return
    
    # Test anomaly detection
    anomalies = agent.detect_anomalies(time_window="1h")
    print(f"Found {len(anomalies)} anomalies")
    
    # Print first anomaly if any
    if anomalies:
        print("Sample anomaly:", anomalies[0])

def test_correlation_agent():
    """Test the correlation agent independently"""
    print("\nTesting Correlation Agent...")
    agent = CorrelationAgent(es_host="http://localhost:9200")
    journeys = agent.track_request_journeys(time_window="1h")
    print(f"Found {len(journeys)} request journeys")

def test_prediction_agent():
    """Test the prediction agent independently"""
    print("\nTesting Prediction Agent...")
    agent = PredictionAgent(es_host="http://localhost:9200")
    for api in ["frontend", "inventory", "payment"]:
        print(f"Generating predictions for {api} API...")
        try:
            prediction = agent.forecast_metrics(api_name=api, hours_back=2)
            if "error" in prediction:
                print(f"Error: {prediction['error']}")
            else:
                print(f"Prediction trend: {prediction.get('prediction_trend')}")
        except Exception as e:
            print(f"Error generating prediction: {e}")

def test_response_agent():
    """Test the response agent independently"""
    print("\nTesting Response Agent...")
    anomaly_agent = AnomalyAgent(es_host="http://localhost:9200") 
    response_agent = ResponseAgent(es_host="http://localhost:9200")
    
    # Get some anomalies to generate recommendations
    anomalies = anomaly_agent.detect_anomalies(time_window="1h")
    if anomalies:
        recommendations = response_agent.generate_recommendations(anomalies)
        print(f"Generated {len(recommendations)} recommendations")
        if recommendations:
            print("Sample recommendation:", recommendations[0])
    else:
        print("No anomalies found to generate recommendations")

if __name__ == "__main__":
    print("Starting agent tests...")
    
    # Run individual tests
    test_anomaly_detection()
    test_correlation_agent()
    test_prediction_agent()
    test_response_agent()
    
    print("\nAll tests completed!")