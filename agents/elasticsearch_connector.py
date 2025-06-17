import os
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import time
import logging
import socket

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ElasticsearchConnector')

class ElasticsearchConnector:
    """A robust connector for Elasticsearch operations used by agents"""
    
    def __init__(self, es_host=None, max_retries=3, retry_interval=5):
        """Initialize the Elasticsearch connector"""
        # First try environment variable
        self.es_host = es_host or os.environ.get('ELASTICSEARCH_HOST')
        
        # If not set, determine if running in Docker or locally
        if not self.es_host:
            # Try Docker service name first
            if self._can_resolve_host('elasticsearch'):
                self.es_host = 'http://elasticsearch:9200'
                logger.info("Using Docker service name for Elasticsearch: %s", self.es_host)
            else:
                # Fall back to localhost
                self.es_host = 'http://localhost:9200'
                logger.info("Using localhost for Elasticsearch: %s", self.es_host)
        
        logger.info(f"Initializing Elasticsearch connector with host: {self.es_host}")
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.es = None
        self._connect()
    
    def _can_resolve_host(self, hostname):
        """Check if a hostname can be resolved"""
        try:
            socket.gethostbyname(hostname)
            return True
        except socket.gaierror:
            return False
    
    def _connect(self):
        """Establish connection to Elasticsearch with retries"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to Elasticsearch at {self.es_host} (attempt {attempt+1}/{self.max_retries})")
                self.es = Elasticsearch([self.es_host], request_timeout=30)
                
                # Test the connection
                if self.es.ping():
                    logger.info("Successfully connected to Elasticsearch")
                    return True
                else:
                    logger.warning("Elasticsearch ping failed")
            except Exception as e:
                logger.error(f"Failed to connect to Elasticsearch: {e}")
            
            # Wait before retrying
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_interval} seconds...")
                time.sleep(self.retry_interval)
        
        logger.error(f"Failed to connect to Elasticsearch after {self.max_retries} attempts")
        return False
    
    def query_logs(self, index_pattern="logs-*", query=None, size=1000, sort_field="@timestamp", sort_order="desc"):
        """Query logs from Elasticsearch"""
        try:
            if not self.es:
                logger.error("Elasticsearch client not initialized")
                return []
            
            if query is None:
                query = {"match_all": {}}
                
            body = {
                "size": size,
                "sort": [{sort_field: {"order": sort_order}}],
                "query": query
            }
            
            # Handle case where index might not exist yet
            if not self.index_exists(index_pattern):
                logger.warning(f"Index {index_pattern} does not exist yet, returning empty result")
                return []
            
            response = self.es.search(index=index_pattern, body=body)
            hits = response.get('hits', {}).get('hits', [])
            
            logger.info(f"Retrieved {len(hits)} documents from Elasticsearch")
            return [hit['_source'] for hit in hits]
        except Exception as e:
            logger.error(f"Error querying Elasticsearch: {e}")
            return []
    
    def index_exists(self, index_pattern):
        """Check if an index exists"""
        try:
            return self.es.indices.exists(index=index_pattern)
        except Exception:
            return False
            
    def query_time_range(self, start_time, end_time, index_pattern="logs-*", additional_query=None, size=1000):
        """Query logs within a specific time range"""
        try:
            if not self.es:
                logger.error("Elasticsearch client not initialized")
                return []
                
            range_query = {
                "range": {
                    "@timestamp": {
                        "gte": start_time,
                        "lte": end_time
                    }
                }
            }
            
            # Combine with additional query if provided
            if additional_query:
                query = {
                    "bool": {
                        "must": [range_query, additional_query]
                    }
                }
            else:
                query = range_query
            
            return self.query_logs(index_pattern=index_pattern, query=query, size=size)
        except Exception as e:
            logger.error(f"Error querying time range: {e}")
            return []
    
    def store_document(self, index, document, doc_id=None):
        """Store a document in Elasticsearch"""
        try:
            if not self.es:
                logger.error("Elasticsearch client not initialized")
                return None
            
            # Add timestamp if not present
            if '@timestamp' not in document:
                document['@timestamp'] = datetime.now().isoformat()
            
            if doc_id:
                result = self.es.index(index=index, id=doc_id, document=document)
            else:
                result = self.es.index(index=index, document=document)
                
            logger.info(f"Document stored in {index} with ID: {result['_id']}")
            return result['_id']
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return None
    
    def store_anomaly(self, anomaly):
        """Store an anomaly detection result"""
        # Use a dedicated index for anomalies
        index = f"anomalies-{datetime.now().strftime('%Y.%m.%d')}"
        
        # Add metadata
        anomaly['stored_at'] = datetime.now().isoformat()
        
        return self.store_document(index, anomaly)
    
    def store_forecast(self, forecast):
        """Store a forecast result"""
        # Use a dedicated index for forecasts
        index = f"forecasts-{datetime.now().strftime('%Y.%m.%d')}"
        
        # Add metadata
        forecast['stored_at'] = datetime.now().isoformat()
        
        return self.store_document(index, forecast)
    
    def store_agent_result(self, agent_type, result):
        """Store any agent analysis result"""
        # Use a dedicated index for agent results
        index = f"agent-results-{datetime.now().strftime('%Y.%m.%d')}"
        
        # Add metadata
        document = {
            'agent_type': agent_type,
            'result': result,
            'created_at': datetime.now().isoformat()
        }
        
        return self.store_document(index, document)
    
    def create_index_template(self):
        """Create index templates for logs, anomalies, and forecasts"""
        try:
            if not self.es:
                logger.error("Elasticsearch client not initialized")
                return False
                
            # Template for logs
            logs_template = {
                "index_patterns": ["logs-*"],
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "api_name": {"type": "keyword"},
                        "environment": {"type": "keyword"},
                        "correlation_id": {"type": "keyword"},
                        "response_time": {"type": "float"},
                        "status_code": {"type": "integer"},
                        "message": {"type": "text"},
                        "log_source": {"type": "keyword"}
                    }
                }
            }
            
            # Template for anomalies
            anomalies_template = {
                "index_patterns": ["anomalies-*"],
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "score": {"type": "float"},
                        "value": {"type": "float"},
                        "is_anomaly": {"type": "boolean"},
                        "context": {"type": "object", "enabled": False},
                        "stored_at": {"type": "date"}
                    }
                }
            }
            
            # Template for forecasts
            forecasts_template = {
                "index_patterns": ["forecasts-*"],
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "api_name": {"type": "keyword"},
                        "current_avg": {"type": "float"},
                        "forecast_avg": {"type": "float"},
                        "trend": {"type": "keyword"},
                        "risk_level": {"type": "keyword"},
                        "forecast_times": {"type": "date"},
                        "forecast_values": {"type": "float"},
                        "stored_at": {"type": "date"}
                    }
                }
            }
            
            # Create the templates
            self.es.indices.put_template(name="logs_template", body=logs_template)
            self.es.indices.put_template(name="anomalies_template", body=anomalies_template)
            self.es.indices.put_template(name="forecasts_template", body=forecasts_template)
            
            logger.info("Index templates created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating index templates: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Test the connector
    es_connector = ElasticsearchConnector()
    
    if es_connector.es and es_connector.es.ping():
        print("Successfully connected to Elasticsearch")
        
        # Create index templates
        es_connector.create_index_template()
        
        # Test query
        logs = es_connector.query_logs(size=5)
        print(f"Retrieved {len(logs)} logs")
        for log in logs[:2]:
            print(json.dumps(log, indent=2))
    else:
        print("Failed to connect to Elasticsearch")