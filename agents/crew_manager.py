from crewai import Agent, Task, Crew
from agents.anomaly_agent import AnomalyAgent
from agents.correlation_agent import CorrelationAgent
from agents.prediction_agent import PredictionAgent
from agents.response_agent import ResponseAgent
import time
import schedule

class MonitoringCrew:
    def __init__(self, es_host="http://localhost:9200", 
                 slack_webhook=None, email_endpoint=None):
        # Initialize agent instances
        self.anomaly_agent = AnomalyAgent(es_host)
        self.correlation_agent = CorrelationAgent(es_host)
        self.prediction_agent = PredictionAgent(es_host)
        self.response_agent = ResponseAgent(es_host, slack_webhook, email_endpoint)
        
        # Create CrewAI agents
        self.agents = {
            "anomaly_detector": Agent(
                role="Anomaly Detection Specialist",
                goal="Detect unusual patterns in API metrics",
                backstory="Expert at identifying outliers and anomalies in time-series data",
                verbose=True,
                allow_delegation=True
            ),
            
            "correlation_expert": Agent(
                role="Request Journey Analyst",
                goal="Track and analyze end-to-end request flows across environments",
                backstory="Specialized in understanding distributed systems and request tracing",
                verbose=True,
                allow_delegation=True
            ),
            
            "forecaster": Agent(
                role="Predictive Analytics Expert",
                goal="Forecast future system behavior and identify potential issues",
                backstory="Data scientist focused on time-series forecasting and trend analysis",
                verbose=True,
                allow_delegation=True
            ),
            
            "responder": Agent(
                role="Incident Response Coordinator",
                goal="Generate actionable recommendations and alert appropriate teams",
                backstory="Experienced in crisis management and system reliability",
                verbose=True,
                allow_delegation=False
            )
        }
        
    def run_monitoring_cycle(self):
        """Executes a complete monitoring cycle with all agents"""
        print("Starting monitoring cycle...")
        
        # Step 1: Detect anomalies
        try:
            anomalies = self.anomaly_agent.detect_anomalies()
            print(f"Detected {len(anomalies)} anomalies")
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            print("Continuing with empty anomaly list")
            anomalies = []
        
        # Step 2: Track request journeys
        journeys = {}
        if anomalies:
            try:
                # Extract correlation IDs from anomalies
                correlation_ids = list(set([
                    a.get('context', {}).get('correlation_id') 
                    for a in anomalies if 'context' in a and 'correlation_id' in a['context']
                ]))
                
                for cid in correlation_ids:
                    if cid:
                        try:
                            result = self.correlation_agent.track_request_journeys(cid)
                            journeys.update(result)
                        except Exception as e:
                            print(f"Error tracking journey for correlation ID {cid}: {str(e)}")
                        
                print(f"Tracked {len(journeys)} request journeys")
            except Exception as e:
                print(f"Error processing request journeys: {str(e)}")
        else:
            print("No anomalies to track request journeys for")
        
        # Step 3: Generate predictions
        predictions = {}
        try:
            for api in ["frontend", "inventory", "payment"]:
                try:
                    prediction = self.prediction_agent.forecast_metrics(api_name=api)
                    if "error" not in prediction:
                        predictions[api] = prediction
                except Exception as e:
                    print(f"Error generating prediction for {api}: {str(e)}")
                    
            print(f"Generated predictions for {len(predictions)} APIs")
        except Exception as e:
            print(f"Error during prediction generation: {str(e)}")
            print("Continuing with empty predictions")
        
        # Step 4: Generate recommendations and alerts
        if anomalies:
            try:
                recommendations = self.response_agent.generate_recommendations(anomalies)
                try:
                    alert_status = self.response_agent.send_alerts(recommendations)
                    print(f"Alert status: {alert_status['status']}")
                except Exception as e:
                    print(f"Error sending alerts: {str(e)}")
                print(f"Generated {len(recommendations)} recommendations")
            except Exception as e:
                print(f"Error generating recommendations: {str(e)}")
        else:
            print("No anomalies detected, skipping recommendations and alerts")
        
        print("Monitoring cycle completed")
        
    def schedule_monitoring(self, interval_minutes=5):
        """Schedule regular monitoring cycles"""
        schedule.every(interval_minutes).minutes.do(self.run_monitoring_cycle)
        
        print(f"Scheduled monitoring every {interval_minutes} minutes")
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    def run_crew_async(self):
        """Run the crew asynchronously using CrewAI tasks and delegation"""
        # Define tasks
        detect_task = Task(
            description="Detect anomalies in API metrics from the last 15 minutes",
            agent=self.agents["anomaly_detector"],
            expected_output="List of detected anomalies with scores and context"
        )
        
        correlate_task = Task(
            description="Analyze request journeys for any anomalous requests",
            agent=self.agents["correlation_expert"],
            expected_output="Complete journey maps for anomalous requests"
        )
        
        forecast_task = Task(
            description="Predict API performance for the next hour based on current trends",
            agent=self.agents["forecaster"],
            expected_output="Forecasts for key metrics with confidence intervals"
        )
        
        respond_task = Task(
            description="Generate recommendations based on anomalies and predictions",
            agent=self.agents["responder"],
            expected_output="Prioritized list of actions and alerts"
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.agents["anomaly_detector"],
                self.agents["correlation_expert"],
                self.agents["forecaster"],
                self.agents["responder"]
            ],
            tasks=[detect_task, correlate_task, forecast_task, respond_task],
            verbose= True,
        )
        
        # Execute crew
        result = crew.kickoff()
        return result

if __name__ == "__main__":
    # Create an instance of the monitoring crew
    print("Starting monitoring crew...")
    monitoring = MonitoringCrew()
    
    # Run a single monitoring cycle for testing
    monitoring.run_monitoring_cycle()
    
    # Uncomment for continuous monitoring
    # monitoring.schedule_monitoring(interval_minutes=5)