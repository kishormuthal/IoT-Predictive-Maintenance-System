"""
Pytest configuration and shared fixtures
Session 1: Foundation & Environment Setup
"""

import pytest
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest environment"""
    # Create reports directory
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Set up test environment
    os.environ['TESTING'] = 'true'
    os.environ['ENVIRONMENT'] = 'test'

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths"""
    for item in items:
        # Add markers based on test file paths
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "dashboard/" in str(item.fspath):
            item.add_marker(pytest.mark.dashboard)
        elif "performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add session markers
        if "session1" in str(item.fspath) or "test_imports" in str(item.fspath):
            item.add_marker(pytest.mark.session1)


# ===== SHARED FIXTURES =====

@pytest.fixture(scope="session")
def project_root_path():
    """Fixture providing project root path"""
    return project_root

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing test data directory"""
    return project_root / 'tests' / 'fixtures'

@pytest.fixture(scope="session")
def sample_sensors():
    """Fixture providing list of 12 NASA sensors for testing"""
    return [
        # SMAP sensors
        "SMAP-ATT-001", "SMAP-COM-001", "SMAP-PAY-001",
        "SMAP-PWR-001", "SMAP-THM-001", "SMAP-SOI-001",
        # MSL sensors
        "MSL-COM-001", "MSL-ENV-001", "MSL-MOB-001",
        "MSL-NAV-001", "MSL-PWR-001", "MSL-SCI-001"
    ]

@pytest.fixture
def sample_sensor_data():
    """Fixture providing sample sensor data for testing"""
    import numpy as np
    import pandas as pd

    # Generate sample time series data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='1min'
    )

    data = {
        'timestamp': timestamps,
        'value': np.random.normal(50, 10, len(timestamps)),
        'equipment_id': 'SMAP-ATT-001',
        'status': 'normal'
    }

    return pd.DataFrame(data)

@pytest.fixture
def mock_equipment_config():
    """Fixture providing mock equipment configuration"""
    return {
        'SMAP-ATT-001': {
            'name': 'Attitude Control System',
            'type': 'attitude',
            'mission': 'SMAP',
            'criticality': 'high'
        },
        'MSL-COM-001': {
            'name': 'Communication System',
            'type': 'communication',
            'mission': 'MSL',
            'criticality': 'medium'
        }
    }

@pytest.fixture
def sample_anomaly_data():
    """Fixture providing sample anomaly data"""
    return {
        'equipment_id': 'SMAP-ATT-001',
        'timestamp': datetime.now(),
        'anomaly_score': 0.95,
        'severity': 'high',
        'description': 'Test anomaly detection',
        'threshold': 0.8
    }

@pytest.fixture
def sample_forecast_data():
    """Fixture providing sample forecast data"""
    return {
        'equipment_id': 'SMAP-ATT-001',
        'forecast_timestamp': datetime.now() + timedelta(hours=1),
        'predicted_value': 55.2,
        'confidence_interval': [50.1, 60.3],
        'model_name': 'transformer_test'
    }

@pytest.fixture
def test_logger():
    """Fixture providing configured test logger"""
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="function")
def clean_environment():
    """Fixture that ensures clean test environment"""
    # Setup
    original_env = os.environ.copy()

    yield

    # Cleanup
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_nasa_data():
    """Fixture providing mock NASA dataset structure"""
    import numpy as np

    return {
        'train': {
            'SMAP-ATT-001': np.random.normal(50, 10, 1000),
            'MSL-COM-001': np.random.normal(30, 5, 1000)
        },
        'test': {
            'SMAP-ATT-001': np.random.normal(50, 10, 200),
            'MSL-COM-001': np.random.normal(30, 5, 200)
        },
        'labels': {
            'SMAP-ATT-001': np.zeros(200),
            'MSL-COM-001': np.zeros(200)
        }
    }

# ===== TEST UTILITIES =====

@pytest.fixture
def import_timeout():
    """Fixture providing import timeout for hanging detection"""
    return 30  # 30 seconds timeout for imports

@pytest.fixture
def performance_thresholds():
    """Fixture providing performance test thresholds"""
    return {
        'import_time': 5.0,  # seconds
        'memory_usage': 500,  # MB
        'response_time': 2.0,  # seconds
        'startup_time': 30.0  # seconds
    }

@pytest.fixture
def test_session_config():
    """Fixture providing test session configuration"""
    return {
        'session_id': 1,
        'session_name': 'Foundation & Environment Setup',
        'test_categories': ['unit', 'import_test', 'core'],
        'expected_coverage': 80.0
    }

# ===== ASYNC FIXTURES =====

@pytest.fixture
def event_loop():
    """Fixture providing event loop for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()