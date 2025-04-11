from crewai import Agent
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class PredictionAgent:
    def __init__(self, es_host="http://localhost:9200"):
        self.es = Elasticsearch([es_host])
    
    def forecast_metrics(self, index_pattern="logstash-*", metric="response_time", 
                         api_name=None, hours_back=24, forecast_horizon=1):
        """
        Forecasts future metrics based on historical data
        """
        query = {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": f"now-{hours_back}h", "lte": "now"}}}
                ]
            }
        }
        
        if api_name:
            query["bool"]["must"].append({"match": {"api_name": api_name}})
            
        response = self.es.search(
            index=index_pattern,
            body={
                "query": query,
                "sort": [{"@timestamp": "asc"}],
                "_source": ["@timestamp", metric, "api_name", "environment"],
                "size": 5000
            }
        )
        
        hits = response['hits']['hits']
        if not hits:
            return {"error": "Not enough data for prediction"}
            
        # Prepare time series data
        df = pd.DataFrame([
            {
                "timestamp": hit['_source'].get('@timestamp'),
                "value": hit['_source'].get(metric, 0),
                "api_name": hit['_source'].get('api_name'),
                "environment": hit['_source'].get('environment')
            }
            for hit in hits
        ])
        
        # Convert to datetime and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Group and resample to regular intervals (5 min buckets)
        if api_name:
            df = df[df['api_name'] == api_name]
            
        df_resampled = df['value'].resample('5min').mean().fillna(method='ffill')
        
        # Simple ARIMA model for forecasting
        try:
            model = ARIMA(df_resampled, order=(2,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12*forecast_horizon)  # 12 steps = 1 hour
            
            # Store prediction
            prediction = {
                "@timestamp": pd.Timestamp.now().isoformat(),
                "api_name": api_name or "all",
                "metric": metric,
                "current_value": float(df_resampled.iloc[-1]),
                "predicted_values": forecast.tolist(),
                "forecast_horizon_hours": forecast_horizon,
                "predicted_mean": float(forecast.mean()),
                "prediction_trend": "increasing" if forecast.mean() > df_resampled.iloc[-1] else "decreasing"
            }
            
            self.es.index(
                index="predictions",
                document=prediction
            )
            
            return prediction
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}