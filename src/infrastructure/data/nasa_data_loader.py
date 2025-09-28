"""
NASA Data Loader
Clean data loading service for SMAP/MSL datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ...core.interfaces.data_interface import DataSourceInterface
from ...core.models.sensor_data import SensorDataBatch, SensorInfo, SensorReading, SensorStatus
from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_by_id

logger = logging.getLogger(__name__)


class NASADataLoader(DataSourceInterface):
    """
    Clean NASA data loading service for 12-sensor configuration
    """

    def __init__(self, data_root: str = "data/raw"):
        """
        Initialize NASA data loader

        Args:
            data_root: Root directory containing NASA datasets
        """
        self.data_root = Path(data_root)
        self.smap_data = None
        self.msl_data = None
        self.labeled_anomalies = None
        self.is_loaded = False

        # Load data on initialization
        self._load_nasa_data()

    def _load_nasa_data(self):
        """Load only the required 12 sensor channels from NASA datasets"""
        try:
            logger.info("Loading NASA datasets for 12 configured sensors...")

            # Get required channels from equipment config
            smap_channels = []
            msl_channels = []

            for equipment in EQUIPMENT_REGISTRY.values():
                if equipment.data_source == "smap":
                    smap_channels.append(equipment.channel_index)
                elif equipment.data_source == "msl":
                    msl_channels.append(equipment.channel_index)

            logger.info(f"Required SMAP channels: {smap_channels}")
            logger.info(f"Required MSL channels: {msl_channels}")

            # Load only required SMAP data
            if smap_channels:
                smap_train_full = np.load(self.data_root / "smap" / "train.npy", allow_pickle=True)
                smap_test_full = np.load(self.data_root / "smap" / "test.npy", allow_pickle=True)
                smap_train_labels = np.load(self.data_root / "smap" / "train_labels.npy", allow_pickle=True)
                smap_test_labels = np.load(self.data_root / "smap" / "test_labels.npy", allow_pickle=True)

                # Extract only required channels
                max_channel = max(smap_channels)
                if max_channel < smap_train_full.shape[1]:
                    smap_train = smap_train_full[:, smap_channels]
                    smap_test = smap_test_full[:, smap_channels]
                else:
                    # Fallback to available channels
                    available_channels = min(len(smap_channels), smap_train_full.shape[1])
                    smap_train = smap_train_full[:, :available_channels]
                    smap_test = smap_test_full[:, :available_channels]

                self.smap_data = {
                    'data': np.vstack([smap_train, smap_test]) if smap_train.size > 0 and smap_test.size > 0 else smap_train,
                    'labels': np.concatenate([smap_train_labels, smap_test_labels]) if smap_train_labels.size > 0 and smap_test_labels.size > 0 else smap_train_labels,
                    'channels': smap_channels
                }
            else:
                self.smap_data = {'data': np.array([]), 'labels': np.array([]), 'channels': []}

            # Load only required MSL data
            if msl_channels:
                msl_train_full = np.load(self.data_root / "msl" / "train.npy", allow_pickle=True)
                msl_test_full = np.load(self.data_root / "msl" / "test.npy", allow_pickle=True)
                msl_train_labels = np.load(self.data_root / "msl" / "train_labels.npy", allow_pickle=True)
                msl_test_labels = np.load(self.data_root / "msl" / "test_labels.npy", allow_pickle=True)

                # Extract only required channels
                max_channel = max(msl_channels)
                if max_channel < msl_train_full.shape[1]:
                    msl_train = msl_train_full[:, msl_channels]
                    msl_test = msl_test_full[:, msl_channels]
                else:
                    # Fallback to available channels
                    available_channels = min(len(msl_channels), msl_train_full.shape[1])
                    msl_train = msl_train_full[:, :available_channels]
                    msl_test = msl_test_full[:, :available_channels]

                self.msl_data = {
                    'data': np.vstack([msl_train, msl_test]) if msl_train.size > 0 and msl_test.size > 0 else msl_train,
                    'labels': np.concatenate([msl_train_labels, msl_test_labels]) if msl_train_labels.size > 0 and msl_test_labels.size > 0 else msl_train_labels,
                    'channels': msl_channels
                }
            else:
                self.msl_data = {'data': np.array([]), 'labels': np.array([]), 'channels': []}

            # Load labeled anomalies if available
            anomaly_file = self.data_root / "labeled_anomalies.csv"
            if anomaly_file.exists():
                self.labeled_anomalies = pd.read_csv(anomaly_file)
            else:
                self.labeled_anomalies = pd.DataFrame()

            self.is_loaded = True

            logger.info(f"NASA data loaded for 12 sensors:")
            logger.info(f"  SMAP data shape: {self.smap_data['data'].shape} (channels: {len(smap_channels)})")
            logger.info(f"  MSL data shape: {self.msl_data['data'].shape} (channels: {len(msl_channels)})")
            logger.info(f"  Total memory footprint reduced significantly")
            logger.info(f"  Labeled anomalies: {len(self.labeled_anomalies)}")

        except Exception as e:
            logger.error(f"Error loading NASA data: {e}")
            self.is_loaded = False
            # Create empty datasets as fallback
            self.smap_data = {'data': np.array([]), 'labels': np.array([]), 'channels': []}
            self.msl_data = {'data': np.array([]), 'labels': np.array([]), 'channels': []}
            self.labeled_anomalies = pd.DataFrame()

    def get_sensor_data(self, sensor_id: str, hours_back: int = 24) -> Dict[str, any]:
        """
        Get sensor data for specified equipment

        Args:
            sensor_id: Equipment ID from EQUIPMENT_REGISTRY
            hours_back: Hours of historical data to retrieve

        Returns:
            Dictionary containing sensor data and metadata
        """
        try:
            # Get equipment configuration
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                raise ValueError(f"Equipment {sensor_id} not found in registry")

            # Get raw data based on data source
            if equipment.data_source == "smap":
                raw_data = self.smap_data['data']
                labels = self.smap_data['labels']
                channels = self.smap_data['channels']
            elif equipment.data_source == "msl":
                raw_data = self.msl_data['data']
                labels = self.msl_data['labels']
                channels = self.msl_data['channels']
            else:
                raise ValueError(f"Unknown data source: {equipment.data_source}")

            if raw_data.size == 0:
                # Return mock data if no real data available
                return self._generate_mock_data(equipment, hours_back)

            # Find the channel index in our loaded data
            try:
                channel_position = channels.index(equipment.channel_index)
                channel_data = raw_data[:, channel_position]
            except (ValueError, IndexError):
                # Channel not found, use first available or generate mock data
                if raw_data.shape[1] > 0:
                    channel_data = raw_data[:, 0]
                    logger.warning(f"Channel {equipment.channel_index} not found for {sensor_id}, using first available channel")
                else:
                    return self._generate_mock_data(equipment, hours_back)

            # Get last N hours of data
            data_points = min(hours_back, len(channel_data))
            values = channel_data[-data_points:]

            # Create timestamps (assuming hourly data)
            end_time = datetime.now()
            timestamps = [
                end_time - timedelta(hours=data_points-i-1)
                for i in range(data_points)
            ]

            # Calculate statistics
            statistics = self._calculate_statistics(values, equipment)

            # Create sensor info
            sensor_info = {
                'sensor_id': sensor_id,
                'name': equipment.name,
                'type': equipment.sensor_type,
                'unit': equipment.unit,
                'equipment_type': equipment.equipment_type.value,
                'criticality': equipment.criticality.value,
                'location': equipment.location,
                'description': equipment.description,
                'normal_range': equipment.normal_range,
                'data_source': equipment.data_source
            }

            return {
                'sensor_id': sensor_id,
                'timestamps': timestamps,
                'values': values.tolist(),
                'sensor_info': sensor_info,
                'statistics': statistics,
                'data_quality': 'real' if self.is_loaded else 'mock'
            }

        except Exception as e:
            logger.error(f"Error getting sensor data for {sensor_id}: {e}")
            # Return mock data as fallback
            equipment = get_equipment_by_id(sensor_id)
            if equipment:
                return self._generate_mock_data(equipment, hours_back)
            else:
                return self._generate_empty_response(sensor_id)

    def _generate_mock_data(self, equipment, hours_back: int) -> Dict[str, any]:
        """Generate mock sensor data when real data is not available"""
        try:
            # Generate realistic data based on equipment configuration
            normal_range = equipment.normal_range
            base_value = (normal_range[0] + normal_range[1]) / 2
            range_width = normal_range[1] - normal_range[0]
            noise_level = range_width * 0.05  # 5% noise

            # Create timestamps
            end_time = datetime.now()
            timestamps = [
                end_time - timedelta(hours=hours_back-i-1)
                for i in range(hours_back)
            ]

            # Generate values with realistic patterns
            values = []
            for i in range(hours_back):
                # Add some periodic pattern + noise
                trend = np.sin(i * 0.1) * range_width * 0.1
                daily_cycle = np.sin(i * 2 * np.pi / 24) * range_width * 0.05
                noise = np.random.normal(0, noise_level)
                value = base_value + trend + daily_cycle + noise

                # Keep within reasonable bounds
                value = max(normal_range[0] * 0.8, min(normal_range[1] * 1.2, value))
                values.append(value)

            # Calculate statistics
            statistics = self._calculate_statistics(np.array(values), equipment)

            # Create sensor info
            sensor_info = {
                'sensor_id': equipment.equipment_id,
                'name': equipment.name,
                'type': equipment.sensor_type,
                'unit': equipment.unit,
                'equipment_type': equipment.equipment_type.value,
                'criticality': equipment.criticality.value,
                'location': equipment.location,
                'description': equipment.description,
                'normal_range': equipment.normal_range,
                'data_source': equipment.data_source
            }

            return {
                'sensor_id': equipment.equipment_id,
                'timestamps': timestamps,
                'values': values,
                'sensor_info': sensor_info,
                'statistics': statistics,
                'data_quality': 'mock'
            }

        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            return self._generate_empty_response(equipment.equipment_id)

    def _generate_empty_response(self, sensor_id: str) -> Dict[str, any]:
        """Generate empty response for error cases"""
        return {
            'sensor_id': sensor_id,
            'timestamps': [],
            'values': [],
            'sensor_info': {'sensor_id': sensor_id, 'name': 'Unknown'},
            'statistics': {},
            'data_quality': 'error'
        }

    def _calculate_statistics(self, values: np.ndarray, equipment) -> Dict[str, any]:
        """Calculate sensor statistics"""
        try:
            if len(values) == 0:
                return {}

            current_value = float(values[-1])
            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            min_value = float(np.min(values))
            max_value = float(np.max(values))

            # Determine status based on thresholds
            if current_value <= equipment.critical_threshold:
                status = SensorStatus.CRITICAL.value
            elif current_value <= equipment.warning_threshold:
                status = SensorStatus.WARNING.value
            elif equipment.normal_range[0] <= current_value <= equipment.normal_range[1]:
                status = SensorStatus.NORMAL.value
            else:
                status = SensorStatus.WARNING.value

            return {
                'current_value': current_value,
                'mean_value': mean_value,
                'std_value': std_value,
                'min_value': min_value,
                'max_value': max_value,
                'status': status,
                'critical_threshold': equipment.critical_threshold,
                'warning_threshold': equipment.warning_threshold,
                'normal_range': equipment.normal_range,
                'data_points': len(values)
            }

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'current_value': 0.0,
                'status': SensorStatus.UNKNOWN.value
            }

    def get_sensor_list(self) -> List[Dict[str, any]]:
        """Get list of available sensors from equipment registry"""
        try:
            sensor_list = []
            for equipment in EQUIPMENT_REGISTRY.values():
                sensor_info = {
                    'sensor_id': equipment.equipment_id,
                    'name': equipment.name,
                    'equipment_type': equipment.equipment_type.value,
                    'sensor_type': equipment.sensor_type,
                    'unit': equipment.unit,
                    'criticality': equipment.criticality.value,
                    'location': equipment.location,
                    'data_source': equipment.data_source,
                    'channel_index': equipment.channel_index
                }
                sensor_list.append(sensor_info)

            logger.info(f"Available sensors: {len(sensor_list)}")
            return sensor_list

        except Exception as e:
            logger.error(f"Error getting sensor list: {e}")
            return []

    def get_latest_value(self, sensor_id: str) -> Dict[str, any]:
        """Get latest sensor reading"""
        try:
            data = self.get_sensor_data(sensor_id, hours_back=1)
            if data['values']:
                return {
                    'sensor_id': sensor_id,
                    'timestamp': data['timestamps'][-1],
                    'value': data['values'][-1],
                    'status': data['statistics'].get('status', 'UNKNOWN'),
                    'unit': data['sensor_info'].get('unit', '')
                }
            else:
                return {
                    'sensor_id': sensor_id,
                    'timestamp': datetime.now(),
                    'value': 0.0,
                    'status': 'UNKNOWN',
                    'unit': ''
                }

        except Exception as e:
            logger.error(f"Error getting latest value for {sensor_id}: {e}")
            return {
                'sensor_id': sensor_id,
                'timestamp': datetime.now(),
                'value': 0.0,
                'status': 'ERROR',
                'unit': ''
            }

    def get_data_quality_report(self) -> Dict[str, any]:
        """Get data quality report"""
        try:
            total_sensors = len(EQUIPMENT_REGISTRY)
            smap_sensors = len([eq for eq in EQUIPMENT_REGISTRY.values() if eq.data_source == "smap"])
            msl_sensors = len([eq for eq in EQUIPMENT_REGISTRY.values() if eq.data_source == "msl"])

            return {
                'total_sensors': total_sensors,
                'smap_sensors': smap_sensors,
                'msl_sensors': msl_sensors,
                'data_loaded': self.is_loaded,
                'smap_data_shape': self.smap_data['data'].shape if self.smap_data['data'].size > 0 else (0, 0),
                'msl_data_shape': self.msl_data['data'].shape if self.msl_data['data'].size > 0 else (0, 0),
                'smap_channels_loaded': len(self.smap_data.get('channels', [])),
                'msl_channels_loaded': len(self.msl_data.get('channels', [])),
                'labeled_anomalies': len(self.labeled_anomalies),
                'data_quality': 'real' if self.is_loaded else 'mock',
                'memory_optimized': True,
                'note': 'Only 12 required sensor channels loaded instead of full dataset'
            }

        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {'error': str(e)}