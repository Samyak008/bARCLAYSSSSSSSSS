from crewai import Agent, Task, Crew
from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
import json
import time
import random
import datetime
import traceback
import os
from dotenv import load_dotenv

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
            
            # 3. Now use CrewAI to analyze these anomalies
            anomaly_task = Task(
                description=f"Analyze these {len(anomalies)} anomalies and identify patterns. "
                           f"What might be causing these unusual API response times? "
                           f"Here are the top anomalies:\n\n{anomaly_details}\n"
                           f"Consider services dependencies, potential bottlenecks, and infrastructure issues.",
                agent=self.agents["anomaly_detector"],
                expected_output="A detailed analysis of the anomaly patterns, potential root causes, and relationships between the anomalies found in the API metrics."
            )
            
            # 4. Create a task for the responder
            response_task = Task(
                description="Generate actionable recommendations to address the detected anomalies. "
                           "Prioritize the actions based on impact and urgency. "
                           "Include specific steps for engineers to follow.",
                agent=self.agents["responder"],
                context=[anomaly_task],  # Use the result from the first task as context
                expected_output="A prioritized list of actionable recommendations with specific steps for engineers to resolve the identified anomalies."
            )
            
            # 5. Create and run the crew with a simpler configuration
            crew = Crew(
                agents=[
                    self.agents["anomaly_detector"],
                    self.agents["responder"]
                ],
                tasks=[anomaly_task, response_task],
                verbose=True,  # Boolean, not integer
                llm=self.llm  # Pass LLM to the crew
            )
            
            # 6. Execute the crew
            try:
                print("\nStarting CrewAI analysis...")
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
    print("Starting file-based CrewAI monitoring system...")
    
    # Create the FileCrew
    crew = FileCrew()
    
    # Run a monitoring cycle
    result = crew.run_monitoring_cycle()
    
    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"crew_analysis_{timestamp}.txt", "w") as f:
        f.write(result)
    
    print(f"\nAnalysis saved to crew_analysis_{timestamp}.txt")

if __name__ == "__main__":
    main()