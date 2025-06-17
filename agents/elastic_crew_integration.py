import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM

from elasticsearch_connector import ElasticsearchConnector
from elastic_anomaly_integration import ElasticsearchAnomalyMonitor

# Load environment variables
load_dotenv()

class ElasticsearchCrewManager:
    """Manages CrewAI agents and integrates them with Elasticsearch"""
    
    def __init__(self, interval=300):  # Default to 5 minute intervals
        """Initialize the Elasticsearch Crew Manager"""
        self.interval = interval  # seconds between crew runs
        self.es_connector = ElasticsearchConnector()
        self.anomaly_monitor = ElasticsearchAnomalyMonitor(interval=interval)
        
        # Initialize LLM
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables")
            self.llm = None
        else:
            self.llm = LLM(
                model="groq/llama3-70b-8192",
                api_key=groq_api_key,
                temperature=0.7,
                stream=True
            )
            print("LLM initialized with Groq")
    
    def _create_crew_agents(self):
        """Create CrewAI agents"""
        if not self.llm:
            print("ERROR: LLM not configured. Agents cannot be created.")
            return None
            
        agents = {
            "anomaly_detector": Agent(
                role="Anomaly Detection Specialist",
                goal="Detect unusual patterns in API metrics and identify their root causes",
                backstory="Expert at identifying outliers and anomalies in time-series data with years of experience in monitoring critical systems",
                verbose=True,
                llm=self.llm
            ),
            
            "correlation_expert": Agent(
                role="Request Journey Analyst",
                goal="Track and analyze end-to-end request flows across environments to find service dependencies",
                backstory="Specialized in understanding distributed systems and request tracing with expertise in microservice architectures",
                verbose=True,
                llm=self.llm
            ),
            
            "forecaster": Agent(
                role="Predictive Analytics Expert",
                goal="Forecast future system behavior and identify potential issues before they become critical",
                backstory="Data scientist focused on time-series forecasting and trend analysis with experience in capacity planning",
                verbose=True,
                llm=self.llm
            ),
            
            "responder": Agent(
                role="Incident Response Coordinator",
                goal="Generate actionable recommendations and plan appropriate responses",
                backstory="Experienced in crisis management, system reliability, and coordinating cross-team responses to incidents",
                verbose=True,
                llm=self.llm
            )
        }
        
        return agents
    
    def format_anomalies_for_crew(self, anomalies, max_anomalies=10):
        """Format anomalies for CrewAI consumption"""
        if not anomalies:
            return "No anomalies detected in this monitoring cycle."
            
        anomaly_details = ""
        sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
        
        for i, anomaly in enumerate(sorted_anomalies[:max_anomalies]):
            anomaly_details += f"Anomaly {i+1}:\n"
            anomaly_details += f"  Score: {anomaly['score']:.2f}\n"
            anomaly_details += f"  Value: {anomaly['value']:.2f} ms\n"
            anomaly_details += f"  API: {anomaly['context'].get('api_name')}\n"
            anomaly_details += f"  Environment: {anomaly['context'].get('environment')}\n"
            if 'status_code' in anomaly['context']:
                anomaly_details += f"  Status Code: {anomaly['context'].get('status_code')}\n"
            if 'timestamp' in anomaly:
                anomaly_details += f"  Timestamp: {anomaly['timestamp']}\n"
            anomaly_details += "\n"
            
        return anomaly_details
    
    def format_forecasts_for_crew(self, forecasts, max_forecasts=5):
        """Format forecasts for CrewAI consumption"""
        if not forecasts:
            return "No forecasts available for this monitoring cycle."
            
        forecast_details = ""
        
        # Sort forecasts by risk level
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for api, forecast in forecasts.items():
            risk_level = forecast.get('risk_level', 'low')
            if risk_level == 'high':
                high_risk.append((api, forecast))
            elif risk_level == 'medium':
                medium_risk.append((api, forecast))
            else:
                low_risk.append((api, forecast))
        
        # Prioritize by risk level
        sorted_forecasts = high_risk + medium_risk + low_risk
        
        # Format the details
        for i, (api, forecast) in enumerate(sorted_forecasts[:max_forecasts]):
            forecast_details += f"API: {api}\n"
            forecast_details += f"  Current Average: {forecast['current_avg']:.2f}ms\n"
            forecast_details += f"  Forecast Average: {forecast['forecast_avg']:.2f}ms\n"
            forecast_details += f"  Trend: {forecast['trend']}\n"
            forecast_details += f"  Risk Level: {forecast.get('risk_level', 'unknown')}\n\n"
            
        return forecast_details
    
    def run_crew_analysis(self, anomalies, forecasts):
        """Run CrewAI analysis on detected anomalies and forecasts"""
        if not self.llm:
            print("ERROR: LLM not configured. CrewAI analysis cannot be performed.")
            return "Error: LLM not configured"
            
        print("Starting CrewAI analysis...")
        
        # Create agents
        agents = self._create_crew_agents()
        if not agents:
            return "Error: Failed to create agents"
            
        # Format data for analysis
        anomaly_details = self.format_anomalies_for_crew(anomalies)
        forecast_details = self.format_forecasts_for_crew(forecasts)
        
        # Create tasks
        anomaly_task = Task(
            description=f"Analyze these anomalies and identify patterns. "
                       f"What might be causing these unusual API response times? "
                       f"Here are the top anomalies:\n\n{anomaly_details}\n"
                       f"Consider services dependencies, potential bottlenecks, and infrastructure issues.",
            agent=agents["anomaly_detector"],
            expected_output="A detailed analysis of the anomaly patterns, potential root causes, and relationships between the anomalies found in the API metrics."
        )
        
        forecast_task = Task(
            description=f"Based on the current trends and forecast data, predict how the system will behave in the next hour. "
                       f"What APIs are likely to experience issues? What's the expected impact?\n\n"
                       f"{forecast_details}\n"
                       f"Consider the anomalies already detected and how the trends might evolve.",
            agent=agents["forecaster"],
            expected_output="A forecast of expected system behavior over the next hour, highlighting potential issues and their likely impact on different services."
        )
        
        response_task = Task(
            description="Based on the anomaly analysis and forecasts, generate actionable recommendations. "
                       "Prioritize the actions based on impact and urgency. "
                       "Include specific steps for engineers to follow.",
            agent=agents["responder"],
            context=[anomaly_task, forecast_task],  # Use results from previous tasks
            expected_output="A prioritized list of actionable recommendations with specific steps for engineers to resolve the identified issues and prevent forecasted problems."
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[
                agents["anomaly_detector"],
                agents["forecaster"],
                agents["responder"]
            ],
            tasks=[anomaly_task, forecast_task, response_task],
            verbose=True,
            llm=self.llm
        )
        
        # Execute the crew
        try:
            result = crew.kickoff()
            print("\nCrewAI Analysis Results:")
            print(result)
            
            # Store the result in Elasticsearch
            self.es_connector.store_agent_result("crew_analysis", result)
            
            return result
        except Exception as e:
            import traceback
            error_msg = f"Error during CrewAI execution: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Store the error in Elasticsearch
            self.es_connector.store_agent_result("crew_error", error_msg)
            
            return error_msg
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle with anomaly detection and CrewAI analysis"""
        print(f"\n=== Starting integrated monitoring cycle at {datetime.now().isoformat()} ===")
        
        # 1. Run anomaly detection to get anomalies and forecasts
        logs = self.anomaly_monitor.get_logs_from_elasticsearch(minutes=30)
        anomalies, forecasts = self.anomaly_monitor.process_logs(logs)
        
        # 2. Run CrewAI analysis on the results
        if anomalies or forecasts:
            analysis_result = self.run_crew_analysis(anomalies, forecasts)
            
            # 3. Store summary in Elasticsearch
            summary = {
                'timestamp': datetime.now().isoformat(),
                'logs_analyzed': len(logs) if logs else 0,
                'anomalies_found': len(anomalies) if anomalies else 0,
                'forecasts_generated': len(forecasts) if forecasts else 0,
                'crew_analysis': analysis_result[:5000] if analysis_result else "No analysis"
            }
            
            self.es_connector.store_document('integrated-monitoring', summary)
        else:
            print("No anomalies or forecasts available for CrewAI analysis")
            
        print("Monitoring cycle complete")
        
    def run_continuous(self):
        """Run monitoring continuously at specified interval"""
        print(f"Starting integrated monitoring (interval: {self.interval}s)")
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
    parser = argparse.ArgumentParser(description='Run integrated monitoring with CrewAI analysis')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    # Create and run the manager
    manager = ElasticsearchCrewManager(interval=args.interval)
    
    if args.once:
        manager.run_once()
    else:
        manager.run_continuous()