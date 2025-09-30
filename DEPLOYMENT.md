# IoT Predictive Maintenance System

## 🚀 Quick Start

### GitHub Codespaces (Recommended)
1. Open this repo in GitHub Codespaces
2. Install dependencies: `pip install -r requirements.txt`
3. Launch dashboard: `python start_dashboard.py`
4. Access at forwarded port 8050

### Local Setup
1. Clone repository
2. Install Python 3.8+
3. Install dependencies: `pip install -r requirements.txt`
4. Train models: `python train_forecasting_models.py --quick`
5. Start dashboard: `python start_dashboard.py`

## 📊 Features
- **Overview**: System metrics and status
- **Monitoring**: Real-time sensor data
- **Anomalies**: Detection and alerts  
- **Forecasting**: Predictive analytics with trained models
- **Maintenance**: Equipment management
- **Work Orders**: Task management
- **Performance**: System health monitoring

## 🔧 Architecture
- **Data**: NASA SMAP/MSL telemetry (12 sensors)
- **Models**: Transformer-based forecasting (219K parameters each)
- **Backend**: Clean Architecture with service layer
- **Frontend**: Dash/Plotly with Bootstrap components
- **ML**: TensorFlow/Scikit-learn for predictions

## 📁 Key Files
- `start_dashboard.py`: Main launcher
- `src/presentation/dashboard/enhanced_app_optimized.py`: Production dashboard
- `train_forecasting_models.py`: Model training pipeline
- `data/models/transformer/`: Trained models directory
