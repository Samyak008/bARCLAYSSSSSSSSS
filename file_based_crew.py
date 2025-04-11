from crewai import Agent, Task, Crew
from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
import json
import time
import random
import datetime
import traceback
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()

class FileCrew:
    def __init__(self):
        # Initialize underlying agents
        self.anomaly_agent = FileBasedAnomalyAgent()
        
        # Configure the LLM with Groq's model - using the format that worked in test_groq.py
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("ERROR: GROQ_API_KEY not found in environment variables.")
            print("Please set it in your .env file.")
            self.llm = None
            return
            
        from crewai import LLM
        self.llm = LLM(
            model="groq/llama3-70b-8192",  # This format works based on the test
            api_key=groq_api_key,
            temperature=0.7,
            stream=True  # Enable streaming for better experience
        )
        print("Groq LLM configured successfully!")
        
        # Create CrewAI agents with Groq LLM
        self.agents = {
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
    
    def analyze_request_journeys(self, logs, anomalies):
        """Extract request journey paths from logs"""
        print("Analyzing request journeys...")
        
        # Group logs by correlation ID to track request journeys
        journeys = {}
        for log in logs:
            cid = log.get('correlation_id')
            if not cid:
                continue
                
            if cid not in journeys:
                journeys[cid] = []
                
            journeys[cid].append(log)
        
        # Focus on journeys containing anomalies
        anomaly_cids = set()
        for anomaly in anomalies:
            cid = anomaly['context'].get('correlation_id')
            if cid:
                anomaly_cids.add(cid)
        
        # Format journeys for analysis
        journey_summaries = []
        for cid, steps in journeys.items():
            if cid in anomaly_cids or len(steps) > 2:  # Focus on anomalous or complex journeys
                # Sort steps by timestamp
                steps.sort(key=lambda x: x.get('@timestamp', ''))
                
                # Create a journey summary
                apis = [step.get('api_name', 'unknown') for step in steps]
                envs = [step.get('environment', 'unknown') for step in steps]
                times = [step.get('response_time', 0) for step in steps]
                statuses = [step.get('status_code', 200) for step in steps]
                
                journey_summaries.append({
                    "correlation_id": cid,
                    "steps": len(steps),
                    "apis": apis,
                    "environments": envs,
                    "response_times": times,
                    "status_codes": statuses,
                    "contains_anomaly": cid in anomaly_cids,
                    "total_time": sum(times),
                    "has_errors": any(status >= 400 for status in statuses)
                })
        
        print(f"Found {len(journey_summaries)} significant request journeys")
        return journey_summaries
    
    def forecast_metrics(self, logs, anomalies):
        """Generate simple forecasts based on historical metrics"""
        print("Generating metric forecasts...")
        
        # Extract timestamp and response_time data
        try:
            # Convert to pandas DataFrame for time series analysis
            df = pd.DataFrame([
                {
                    "timestamp": log.get('@timestamp'),
                    "response_time": log.get('response_time', 0),
                    "api_name": log.get('api_name', 'unknown'),
                    "environment": log.get('environment', 'unknown')
                }
                for log in logs
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Group by API name
            api_names = df['api_name'].unique()
            forecasts = {}
            
            for api in api_names:
                api_data = df[df['api_name'] == api]['response_time']
                
                if len(api_data) > 10:  # Only forecast if we have enough data
                    try:
                        # Resample to regular intervals
                        resampled = api_data.resample('1S').mean().fillna(method='ffill')
                        
                        # Simple ARIMA model
                        model = ARIMA(resampled.values, order=(2, 1, 0))
                        model_fit = model.fit()
                        
                        # Forecast next 5 minutes
                        forecast = model_fit.forecast(steps=5)
                        current_avg = api_data.mean()
                        forecast_avg = forecast.mean()
                        trend = "increasing" if forecast_avg > current_avg else "decreasing"
                        
                        forecasts[api] = {
                            "current_avg": current_avg,
                            "forecast_avg": forecast_avg,
                            "trend": trend,
                            "forecast_values": forecast.tolist()
                        }
                    except Exception as e:
                        # Fall back to simple average-based forecast
                        current_avg = api_data.mean()
                        forecasts[api] = {
                            "current_avg": current_avg,
                            "forecast_avg": current_avg,
                            "trend": "stable",
                            "note": f"ARIMA failed, using average: {str(e)}"
                        }
            
            print(f"Generated forecasts for {len(forecasts)} APIs")
            return forecasts
            
        except Exception as e:
            print(f"Error in forecasting: {str(e)}")
            return {}
        
    def run_monitoring_cycle(self, log_file='api_logs.json'):
        """Run a complete monitoring cycle with CrewAI agents"""
        print("Starting CrewAI monitoring cycle...")
        
        # Check if LLM is configured
        if not self.llm:
            return "ERROR: LLM not configured. Please check your Groq API key."
        
        try:
            # 1. Load logs
            logs = self.anomaly_agent.load_logs_from_file(log_file)
            if not logs:
                print("No logs found. Please generate some logs first.")
                return "No logs available for analysis"
            
            # 2. Detect anomalies using the file-based agent
            anomalies, _, _ = self.anomaly_agent.detect_anomalies(logs, threshold=0.8)
            print(f"Detected {len(anomalies)} anomalies")
            
            # 3. Analyze request journeys
            journeys = self.analyze_request_journeys(logs, anomalies)
            
            # 4. Generate metric forecasts
            forecasts = self.forecast_metrics(logs, anomalies)
            
            # Prepare anomaly information for the agents
            anomaly_details = ""
            if anomalies:
                sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
                for i, anomaly in enumerate(sorted_anomalies[:10]):
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
            else:
                anomaly_details = "No significant anomalies detected in this monitoring cycle."
            
            # Format journey information
            journey_details = ""
            for i, journey in enumerate(journeys[:5]):
                journey_details += f"Journey {i+1} (ID: {journey['correlation_id']}):\n"
                journey_details += f"  Steps: {journey['steps']}\n"
                journey_details += f"  APIs: {' → '.join(journey['apis'])}\n"
                journey_details += f"  Environments: {', '.join(set(journey['environments']))}\n"
                journey_details += f"  Total Time: {journey['total_time']}ms\n"
                journey_details += f"  Contains Anomalies: {'Yes' if journey['contains_anomaly'] else 'No'}\n"
                journey_details += f"  Contains Errors: {'Yes' if journey['has_errors'] else 'No'}\n\n"
            
            # Format forecast information
            forecast_details = ""
            for api, forecast in forecasts.items():
                forecast_details += f"API: {api}\n"
                forecast_details += f"  Current Average: {forecast['current_avg']:.2f}ms\n"
                forecast_details += f"  Forecast Average: {forecast['forecast_avg']:.2f}ms\n"
                forecast_details += f"  Trend: {forecast['trend']}\n\n"
            
            # 5. Define CrewAI tasks
            
            # Task 1: Anomaly detection and analysis
            anomaly_task = Task(
                description=f"Analyze these {len(anomalies)} anomalies and identify patterns. "
                           f"What might be causing these unusual API response times? "
                           f"Here are the top anomalies:\n\n{anomaly_details}\n"
                           f"Consider services dependencies, potential bottlenecks, and infrastructure issues.",
                agent=self.agents["anomaly_detector"],
                expected_output="A detailed analysis of the anomaly patterns, potential root causes, and relationships between the anomalies found in the API metrics."
            )
            
            # Task 2: Correlation analysis
            correlation_task = Task(
                description=f"Analyze these request journeys and identify patterns and dependencies between services. "
                           f"What service dependencies can be inferred? Are there any bottlenecks or problematic transitions?\n\n"
                           f"{journey_details}\n"
                           f"Consider how requests flow between services and environments, and identify any problematic patterns.",
                agent=self.agents["correlation_expert"],
                expected_output="An analysis of service dependencies, request flows, and identification of bottlenecks or problematic transitions between services."
            )
            
            # Task 3: Forecasting
            forecast_task = Task(
                description=f"Based on the current trends and forecast data, predict how the system will behave in the next hour. "
                           f"What APIs are likely to experience issues? What's the expected impact?\n\n"
                           f"{forecast_details}\n"
                           f"Consider the anomalies already detected and how the trends might evolve.",
                agent=self.agents["forecaster"],
                expected_output="A forecast of expected system behavior over the next hour, highlighting potential issues and their likely impact on different services."
            )
            
            # Task 4: Response recommendations
            response_task = Task(
                description="Based on the anomaly analysis, correlation patterns, and forecasts, generate actionable recommendations. "
                           "Prioritize the actions based on impact and urgency. "
                           "Include specific steps for engineers to follow.",
                agent=self.agents["responder"],
                context=[anomaly_task, correlation_task, forecast_task],  # Use results from all previous tasks
                expected_output="A prioritized list of actionable recommendations with specific steps for engineers to resolve the identified issues and prevent forecasted problems."
            )
            
            # 6. Create and run the crew with all agents
            crew = Crew(
                agents=[
                    self.agents["anomaly_detector"],
                    self.agents["correlation_expert"],
                    self.agents["forecaster"],
                    self.agents["responder"]
                ],
                tasks=[anomaly_task, correlation_task, forecast_task, response_task],
                verbose=True,
                llm=self.llm
            )
            
            # 7. Execute the crew
            try:
                print("\nStarting CrewAI analysis with all agents...")
                result = crew.kickoff()
                print("\nCrew Analysis Results:")
                print(result)
                return result
            except Exception as e:
                error_msg = f"Error during CrewAI execution: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error during monitoring cycle: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

def main():
    print("Starting complete file-based CrewAI monitoring system...")
    
    # Create the FileCrew
    crew = FileCrew()
    
    # Run a monitoring cycle
    result = crew.run_monitoring_cycle()
    
    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"complete_crew_analysis_{timestamp}.txt", "w") as f:
        f.write(result)
    
    print(f"\nAnalysis saved to complete_crew_analysis_{timestamp}.txt")

if __name__ == "__main__":
    main()