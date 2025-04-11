import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime, timedelta

class EnhancedForecastingAgent:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.anomalies = []
        
    def load_logs_from_file(self, filename, limit=5000):
        """Load logs from a JSON file"""
        logs = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line.strip()))
                        if len(logs) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading logs: {e}")
        
        print(f"Loaded {len(logs)} logs from {filename}")
        return logs
    
    def prepare_time_series(self, logs, metric="response_time"):
        """Convert logs to time series format by API"""
        if not logs:
            return None
            
        # Convert to dataframe
        data = []
        for log in logs:
            data.append({
                'timestamp': log.get('@timestamp'),
                'api_name': log.get('api_name', 'unknown'),
                'value': log.get(metric, 0),
                'environment': log.get('environment', 'unknown'),
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Group by API
        api_series = {}
        for api in df['api_name'].unique():
            api_data = df[df['api_name'] == api]['value']
            # Resample to regular intervals
            api_resampled = api_data.resample('1S').mean().fillna(method='ffill')
            api_series[api] = api_resampled
            
        return api_series, df
    
    def incorporate_anomalies(self, anomalies, api_series):
        """Use anomaly information to improve forecasting"""
        self.anomalies = anomalies
        
        # Create anomaly indicators by API
        anomaly_indicators = {}
        if anomalies:
            for api in api_series.keys():
                # Initialize a series of zeros matching our time series
                indicators = pd.Series(0, index=api_series[api].index)
                
                # Mark anomaly points
                for anomaly in anomalies:
                    if anomaly['context'].get('api_name') == api:
                        try:
                            # Use the timestamp to mark this point
                            timestamp = pd.to_datetime(anomaly['timestamp'])
                            if timestamp in indicators.index:
                                indicators[timestamp] = 1
                        except (ValueError, TypeError):
                            pass
                
                anomaly_indicators[api] = indicators
        
        return anomaly_indicators
    
    def select_best_model(self, api, train_data, test_data=None):
        """Select the best forecasting model based on validation performance"""
        if len(train_data) < 30:
            print(f"Not enough data for {api}, using simple exponential smoothing")
            return "exponential", None
        
        train_size = int(len(train_data) * 0.8)
        if test_data is None:
            # Use part of train_data as validation
            validation_data = train_data[train_size:]
            train_data = train_data[:train_size]
        else:
            validation_data = test_data
        
        print(f"Selecting best model for {api} with {len(train_data)} points")
        
        # Define models to try
        models = {
            "arima": ARIMA(train_data, order=(2, 1, 2)),
            "sarima": SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 60)),
            "exponential": ExponentialSmoothing(train_data, trend='add', seasonal=None)
        }
        
        # Evaluate each model
        errors = {}
        forecasts = {}
        
        for name, model in models.items():
            try:
                print(f"  Fitting {name} model...")
                model_fit = model.fit(disp=0)
                forecast = model_fit.forecast(steps=len(validation_data))
                mse = ((forecast - validation_data) ** 2).mean()
                errors[name] = mse
                forecasts[name] = forecast
                print(f"  {name} MSE: {mse:.2f}")
            except Exception as e:
                print(f"  {name} failed: {str(e)}")
                errors[name] = float('inf')
        
        # Select best model
        best_model = min(errors, key=errors.get)
        print(f"Best model for {api}: {best_model} (MSE: {errors[best_model]:.2f})")
        
        return best_model, forecasts[best_model] if best_model in forecasts else None
    
    def generate_forecasts(self, logs, anomalies=None, horizon_minutes=30):
        """Generate forecasts for all APIs"""
        print("Generating enhanced forecasts...")
        
        # Prepare data
        api_series, full_df = self.prepare_time_series(logs)
        
        if api_series is None:
            return {}
        
        # Incorporate anomaly information
        anomaly_indicators = self.incorporate_anomalies(anomalies, api_series)
        
        # Generate forecasts for each API
        results = {}
        for api, series in api_series.items():
            if len(series) < 15:  # Skip APIs with too little data
                continue
                
            print(f"\nForecasting {api} with {len(series)} data points...")
            
            # Select best model
            model_type, _ = self.select_best_model(api, series)
            
            # Train on full data
            try:
                if model_type == "arima":
                    model = ARIMA(series, order=(2, 1, 2))
                elif model_type == "sarima":
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 0, 60))
                else:  # exponential
                    model = ExponentialSmoothing(series, trend='add', seasonal=None)
                
                model_fit = model.fit(disp=0)
                
                # Forecast future values
                steps = horizon_minutes * 60  # Convert minutes to seconds
                forecast = model_fit.forecast(steps=steps)
                
                # Store results
                last_timestamp = series.index[-1]
                forecast_times = pd.date_range(start=last_timestamp, periods=steps+1, freq='S')[1:]
                
                forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
                
                current_avg = series.iloc[-60:].mean() if len(series) > 60 else series.mean()
                forecast_avg = forecast.mean()
                
                # Calculate prediction intervals (only for ARIMA/SARIMA)
                confidence_lower = None
                confidence_upper = None
                
                if model_type in ["arima", "sarima"]:
                    try:
                        # Get prediction intervals
                        pred_int = model_fit.get_forecast(steps=steps).conf_int(alpha=0.2)
                        confidence_lower = pred_int.iloc[:, 0].values
                        confidence_upper = pred_int.iloc[:, 1].values
                    except Exception as e:
                        print(f"  Error calculating confidence intervals: {str(e)}")
                
                # Determine trend
                if forecast_avg > current_avg * 1.2:
                    trend = "strongly_increasing"
                elif forecast_avg > current_avg * 1.05:
                    trend = "increasing"
                elif forecast_avg < current_avg * 0.8:
                    trend = "strongly_decreasing"
                elif forecast_avg < current_avg * 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                results[api] = {
                    "current_avg": float(current_avg),
                    "forecast_avg": float(forecast_avg),
                    "trend": trend,
                    "model_type": model_type,
                    "forecast_times": [t.isoformat() for t in forecast_times],
                    "forecast_values": [float(v) for v in forecast_values],
                    "confidence_lower": [float(v) for v in confidence_lower] if confidence_lower is not None else None,
                    "confidence_upper": [float(v) for v in confidence_upper] if confidence_upper is not None else None,
                    "anomaly_history": anomalies is not None and any(a['context'].get('api_name') == api for a in anomalies)
                }
                
                # Calculate potential anomaly risk
                if results[api]["anomaly_history"] and trend in ["strongly_increasing", "increasing"]:
                    results[api]["risk_level"] = "high"
                elif results[api]["anomaly_history"] or trend in ["strongly_increasing"]:
                    results[api]["risk_level"] = "medium"
                else:
                    results[api]["risk_level"] = "low"
                
            except Exception as e:
                print(f"Error forecasting {api}: {str(e)}")
        
        # Plot the forecasts
        self.plot_forecasts(api_series, results)
        
        return results
    
    def plot_forecasts(self, api_series, forecasts):
        """Create visualization of forecasts for each API"""
        if not forecasts:
            return
            
        n_apis = len(forecasts)
        if n_apis > 9:
            # Just plot a selection of APIs with most data
            apis_to_plot = sorted(api_series.keys(), key=lambda a: len(api_series[a]), reverse=True)[:9]
        else:
            apis_to_plot = list(forecasts.keys())
        
        # Calculate grid dimensions
        n_cols = min(3, n_apis)
        n_rows = (len(apis_to_plot) + n_cols - 1) // n_cols
        
        # Create plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4), squeeze=False)
        
        for i, api in enumerate(apis_to_plot):
            if api not in forecasts:
                continue
                
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot historical data
            series = api_series[api]
            ax.plot(series.index, series.values, 'b-', label='Historical', alpha=0.7)
            
            # Plot forecast
            forecast = forecasts[api]
            forecast_times = [datetime.fromisoformat(t) for t in forecast['forecast_times']]
            forecast_values = forecast['forecast_values']
            
            ax.plot(forecast_times, forecast_values, 'r-', label='Forecast')
            
            # Plot confidence intervals if available
            if forecast['confidence_lower'] is not None and forecast['confidence_upper'] is not None:
                ax.fill_between(
                    forecast_times,
                    forecast['confidence_lower'],
                    forecast['confidence_upper'],
                    color='r', alpha=0.2, label='80% Confidence'
                )
            
            # Mark anomalies if available
            for anomaly in self.anomalies:
                if anomaly['context'].get('api_name') == api:
                    try:
                        timestamp = datetime.fromisoformat(anomaly['timestamp'])
                        if timestamp in series.index:
                            ax.scatter([timestamp], [series.loc[timestamp]], 
                                      color='red', s=50, marker='x')
                    except (ValueError, TypeError, KeyError):
                        pass
            
            # Set title and labels
            risk = forecast.get('risk_level', 'unknown')
            risk_color = {'high': 'red', 'medium': 'orange', 'low': 'green'}.get(risk, 'black')
            ax.set_title(f"{api} - {forecast['trend']} (Risk: {risk})", color=risk_color)
            ax.set_ylabel('Response Time')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(apis_to_plot), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('forecast_predictions.png')
        print("Forecast visualization saved to forecast_predictions.png")

def main():
    # Test the enhanced forecasting
    agent = EnhancedForecastingAgent()
    
    # Load logs
    logs = agent.load_logs_from_file('api_logs.json')
    
    if logs:
        # Generate forecasts without anomaly information
        print("\nGenerating forecasts without anomaly information:")
        forecasts1 = agent.generate_forecasts(logs)
        
        # Try loading anomalies if available
        try:
            from enhanced_anomaly_agent import EnhancedAnomalyAgent
            anomaly_agent = EnhancedAnomalyAgent()
            anomalies, _, _ = anomaly_agent.detect_anomalies(logs)
            
            # Generate improved forecasts with anomaly information
            print("\nGenerating forecasts with anomaly information:")
            forecasts2 = agent.generate_forecasts(logs, anomalies)
            
            # Compare results
            print("\nComparison of forecasts with and without anomaly information:")
            for api in forecasts2:
                if api in forecasts1:
                    print(f"{api}: ")
                    print(f"  Without anomalies: {forecasts1[api]['trend']} (Risk: {forecasts1[api].get('risk_level', 'N/A')})")
                    print(f"  With anomalies: {forecasts2[api]['trend']} (Risk: {forecasts2[api].get('risk_level', 'N/A')})")
                    
        except (ImportError, Exception) as e:
            print(f"Could not load anomaly information: {str(e)}")

if __name__ == "__main__":
    main()