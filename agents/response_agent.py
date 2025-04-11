from crewai import Agent, Task, Crew
from elasticsearch import Elasticsearch
import requests
import json

class ResponseAgent:
    def __init__(self, es_host="http://elasticsearch:9200", 
                 slack_webhook=None, email_endpoint=None):
        self.es = Elasticsearch([es_host])
        self.slack_webhook = slack_webhook
        self.email_endpoint = email_endpoint
    
    def generate_recommendations(self, anomaly_data):
        """
        Generates response recommendations based on anomalies and predictions
        """
        if not anomaly_data:
            return []
            
        recommendations = []
        for anomaly in anomaly_data:
            context = anomaly.get('context', {})
            api_name = context.get('api_name', 'unknown')
            environment = context.get('environment', 'unknown')
            metric_value = anomaly.get('value')
            score = anomaly.get('score', 0)
            
            # Basic recommendation logic
            if score > 0.95:  # Critical anomaly
                action = f"Critical: Investigate {api_name} API in {environment} environment immediately"
                recommendations.append({
                    "severity": "critical",
                    "action": action,
                    "context": context,
                    "details": f"Extremely abnormal response time of {metric_value}ms detected"
                })
            elif score > 0.8:  # High anomaly
                action = f"High: Scale {api_name} API instances in {environment} environment"
                recommendations.append({
                    "severity": "high",
                    "action": action,
                    "context": context,
                    "details": f"Abnormal response time of {metric_value}ms detected"
                })
            else:  # Moderate anomaly
                action = f"Monitor {api_name} API in {environment} environment closely"
                recommendations.append({
                    "severity": "moderate",
                    "action": action,
                    "context": context,
                    "details": f"Unusual response time pattern detected: {metric_value}ms"
                })
        
        # Store recommendations
        if recommendations:
            self.es.index(
                index="recommendations",
                document={
                    "@timestamp": anomaly_data[0].get('timestamp') if anomaly_data else None,
                    "recommendations": recommendations,
                    "num_anomalies": len(anomaly_data)
                }
            )
            
        return recommendations
    
    def send_alerts(self, recommendations):
        """
        Sends alerts based on recommendations
        """
        if not recommendations:
            return {"status": "no alerts needed"}
            
        # Filter for important alerts
        critical_alerts = [r for r in recommendations if r.get('severity') == 'critical']
        high_alerts = [r for r in recommendations if r.get('severity') == 'high']
        
        alerts_sent = []
        
        # Send Slack alerts for critical/high issues
        if self.slack_webhook and (critical_alerts or high_alerts):
            alert_text = "🚨 *API MONITORING ALERT* 🚨\n\n"
            
            for alert in critical_alerts:
                alert_text += f"*CRITICAL:* {alert['action']}\n{alert['details']}\n\n"
                
            for alert in high_alerts[:3]:  # Limit to top 3 high alerts
                alert_text += f"*HIGH:* {alert['action']}\n{alert['details']}\n\n"
                
            try:
                response = requests.post(
                    self.slack_webhook,
                    data=json.dumps({"text": alert_text}),
                    headers={"Content-Type": "application/json"}
                )
                alerts_sent.append({
                    "channel": "slack",
                    "status": "sent" if response.status_code == 200 else "failed",
                    "alerts_count": len(critical_alerts) + len(high_alerts[:3])
                })
            except Exception as e:
                alerts_sent.append({
                    "channel": "slack", 
                    "status": "error", 
                    "message": str(e)
                })
                
        return {
            "status": "alerts sent" if alerts_sent else "no alerts sent",
            "details": alerts_sent
        }