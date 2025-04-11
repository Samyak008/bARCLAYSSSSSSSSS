from crewai import Agent
from elasticsearch import Elasticsearch

class CorrelationAgent:
    def __init__(self, es_host="http://elasticsearch:9200"):
        self.es = Elasticsearch([es_host])
    
    def track_request_journeys(self, correlation_id=None, time_window="1h"):
        """
        Tracks complete request journeys across distributed environments
        by following correlation IDs
        """
        query = {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": f"now-{time_window}", "lte": "now"}}}
                ]
            }
        }
        
        if correlation_id:
            query["bool"]["must"].append({"match": {"correlation_id": correlation_id}})
            
        response = self.es.search(
            index="logstash-*",
            body={
                "query": query,
                "sort": [{"@timestamp": "asc"}],
                "_source": ["@timestamp", "api_name", "environment", 
                           "correlation_id", "status_code", "response_time"],
                "size": 1000
            }
        )
        
        # Group by correlation IDs
        journeys = {}
        for hit in response['hits']['hits']:
            source = hit['_source']
            cid = source.get('correlation_id', 'unknown')
            
            if cid not in journeys:
                journeys[cid] = []
                
            journeys[cid].append({
                "timestamp": source.get('@timestamp'),
                "api_name": source.get('api_name'),
                "environment": source.get('environment'),
                "status_code": source.get('status_code'),
                "response_time": source.get('response_time'),
            })
        
        # Calculate journey metrics and store enriched journeys
        for cid, steps in journeys.items():
            if len(steps) > 1:  # Only process multi-step journeys
                total_time = sum(step.get('response_time', 0) for step in steps)
                environments = [step['environment'] for step in steps]
                has_errors = any(step.get('status_code', 200) >= 400 for step in steps)
                
                # Store journey summary
                self.es.index(
                    index="journeys",
                    document={
                        "@timestamp": steps[0]['timestamp'],
                        "correlation_id": cid,
                        "journey_steps": len(steps),
                        "environments": list(set(environments)),
                        "total_time": total_time,
                        "has_errors": has_errors,
                        "steps": steps
                    }
                )
                
        return journeys