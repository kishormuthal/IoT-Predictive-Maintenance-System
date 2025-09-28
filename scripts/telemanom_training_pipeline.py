#!/usr/bin/env python3
"""
Telemanom Training Pipeline
Separate training pipeline for updating Telemanom models for 12-equipment configuration
This pipeline is independent of the dashboard and focuses only on model training
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import configuration and services
from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_list
from src.data_ingestion.real_data_service import get_real_data_service
from src.anomaly_detection.telemanom import NASATelemanom, Telemanom_Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemanamTrainingPipeline:
    """Training pipeline for Telemanom models"""

    def __init__(self, models_dir: str = "data/models"):
        """Initialize training pipeline"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.data_service = get_real_data_service()
        self.equipment_registry = EQUIPMENT_REGISTRY

        # Training statistics
        self.training_stats = {
            'total_equipment': len(self.equipment_registry),
            'trained_models': 0,
            'failed_training': 0,
            'training_start': None,
            'training_end': None,
            'results': {}
        }

        logger.info(f"Telemanom Training Pipeline initialized for {len(self.equipment_registry)} equipment")

    def train_all_equipment(self, force_retrain: bool = False):
        """Train Telemanom models for all equipment"""
        logger.info("Starting training for all equipment models...")
        self.training_stats['training_start'] = datetime.now()

        for equipment_id in self.equipment_registry.keys():
            try:
                logger.info(f"Training model for equipment: {equipment_id}")
                result = self.train_equipment_model(equipment_id, force_retrain)
                self.training_stats['results'][equipment_id] = result

                if result['success']:
                    self.training_stats['trained_models'] += 1
                else:
                    self.training_stats['failed_training'] += 1

            except Exception as e:
                logger.error(f"Error training {equipment_id}: {e}")
                self.training_stats['failed_training'] += 1
                self.training_stats['results'][equipment_id] = {
                    'success': False,
                    'error': str(e),
                    'training_time': 0
                }

        self.training_stats['training_end'] = datetime.now()
        self._save_training_report()

        logger.info(f"Training completed: {self.training_stats['trained_models']} successful, "
                   f"{self.training_stats['failed_training']} failed")

    def train_equipment_model(self, equipment_id: str, force_retrain: bool = False):
        """Train Telemanom model for specific equipment"""
        try:
            # Check if model already exists
            model_path = self.models_dir / f"{equipment_id}_anomaly_detector_best.h5"
            if model_path.exists() and not force_retrain:
                logger.info(f"Model for {equipment_id} already exists, skipping (use --force to retrain)")
                return {
                    'success': True,
                    'action': 'skipped',
                    'reason': 'model_exists',
                    'training_time': 0
                }

            # Get equipment configuration
            equipment_config = self.equipment_registry[equipment_id]

            # Prepare training data
            training_data = self._prepare_training_data(equipment_id)

            if training_data is None or len(training_data) < 100:
                raise ValueError(f"Insufficient training data for {equipment_id}")

            # Create Telemanom configuration
            telemanom_config = self._create_telemanom_config(equipment_id)

            # Initialize and train model
            start_time = datetime.now()
            model = NASATelemanom(sensor_id=equipment_id, config=telemanom_config)

            # Train the model
            training_result = model.fit(training_data)

            # Save the trained model
            model_save_path = self.models_dir / f"{equipment_id}_anomaly_detector_best.h5"
            model.save_model(str(model_save_path))

            # Save metadata
            metadata = self._create_model_metadata(equipment_id, telemanom_config, training_result, start_time)
            metadata_path = self.models_dir / f"{equipment_id}_metadata.json"

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            logger.info(f"Successfully trained model for {equipment_id} in {training_time:.2f} seconds")

            return {
                'success': True,
                'action': 'trained',
                'training_time': training_time,
                'final_loss': training_result.get('final_loss', 0),
                'epochs': training_result.get('epochs', 0),
                'model_path': str(model_save_path),
                'metadata_path': str(metadata_path)
            }

        except Exception as e:
            logger.error(f"Error training model for {equipment_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': 0
            }

    def _prepare_training_data(self, equipment_id: str):
        """Prepare training data for equipment"""
        try:
            # Get extended historical data for training
            equipment_data = self.data_service.get_equipment_data(equipment_id, hours_back=168)  # 1 week

            # Extract values and normalize
            values = equipment_data['values']

            if len(values) < 100:
                logger.warning(f"Insufficient data for {equipment_id}: only {len(values)} points")
                return None

            # Simple normalization (can be enhanced)
            normalized_values = self._normalize_data(values)

            # Reshape for Telemanom (expects 2D array)
            training_data = normalized_values.reshape(-1, 1)

            logger.info(f"Prepared training data for {equipment_id}: {training_data.shape}")
            return training_data

        except Exception as e:
            logger.error(f"Error preparing training data for {equipment_id}: {e}")
            return None

    def _normalize_data(self, data):
        """Normalize data for training"""
        try:
            data = np.array(data)

            # Remove outliers (simple approach)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)

            # Clip outliers
            data = np.clip(data, lower_bound, upper_bound)

            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)

            if std > 0:
                normalized = (data - mean) / std
            else:
                normalized = data - mean

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data

    def _create_telemanom_config(self, equipment_id: str):
        """Create Telemanom configuration for equipment"""
        # Get equipment-specific configuration
        equipment_config = self.equipment_registry[equipment_id]

        # Create optimized configuration based on equipment type and criticality
        config = Telemanom_Config()

        # Adjust parameters based on criticality
        if equipment_config.criticality.value == 'CRITICAL':
            config.sequence_length = 60  # Longer sequences for critical equipment
            config.encoder_units = [64, 32]
            config.latent_dim = 16
            config.epochs = 50
        elif equipment_config.criticality.value == 'HIGH':
            config.sequence_length = 50
            config.encoder_units = [48, 24]
            config.latent_dim = 12
            config.epochs = 40
        else:
            config.sequence_length = 40
            config.encoder_units = [32, 16]
            config.latent_dim = 8
            config.epochs = 35

        # Common parameters
        config.batch_size = 32
        config.learning_rate = 0.001
        config.validation_split = 0.2

        return config

    def _create_model_metadata(self, equipment_id: str, config, training_result, start_time):
        """Create model metadata"""
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        metadata = {
            "equipment_id": equipment_id,
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "sequence_length": config.sequence_length,
                "encoder_units": config.encoder_units,
                "latent_dim": config.latent_dim,
                "learning_rate": config.learning_rate,
                "validation_split": config.validation_split
            },
            "metrics": {
                "training_time": training_time,
                "final_epoch": training_result.get('epochs', 0),
                "best_train_loss": training_result.get('final_loss', 0),
                "best_val_loss": training_result.get('val_loss', 0),
                "convergence_ratio": training_result.get('convergence_ratio', 0),
                "training_samples": training_result.get('training_samples', 0),
                "validation_samples": training_result.get('validation_samples', 0)
            },
            "training_date": datetime.now().isoformat(),
            "equipment_info": {
                "name": self.equipment_registry[equipment_id].name,
                "type": self.equipment_registry[equipment_id].equipment_type.value,
                "criticality": self.equipment_registry[equipment_id].criticality.value,
                "location": self.equipment_registry[equipment_id].location
            },
            "scaler_params": {
                "scale_": None,  # Would store actual scaler parameters if using sklearn scaler
                "min_": None,
                "data_min_": None,
                "data_max_": None
            }
        }

        return metadata

    def _save_training_report(self):
        """Save comprehensive training report"""
        report = {
            "training_summary": self.training_stats,
            "equipment_results": self.training_stats['results'],
            "total_duration": (
                self.training_stats['training_end'] - self.training_stats['training_start']
            ).total_seconds() if self.training_stats['training_end'] else 0,
            "success_rate": (
                self.training_stats['trained_models'] / self.training_stats['total_equipment']
            ) * 100 if self.training_stats['total_equipment'] > 0 else 0
        }

        report_path = self.models_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved to: {report_path}")

    def validate_models(self):
        """Validate all trained models"""
        logger.info("Validating trained models...")

        validation_results = {}

        for equipment_id in self.equipment_registry.keys():
            try:
                model_path = self.models_dir / f"{equipment_id}_anomaly_detector_best.h5"
                metadata_path = self.models_dir / f"{equipment_id}_metadata.json"

                if not model_path.exists():
                    validation_results[equipment_id] = {
                        'valid': False,
                        'error': 'Model file missing'
                    }
                    continue

                if not metadata_path.exists():
                    validation_results[equipment_id] = {
                        'valid': False,
                        'error': 'Metadata file missing'
                    }
                    continue

                # Load and validate metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Check model file size
                model_size = model_path.stat().st_size

                validation_results[equipment_id] = {
                    'valid': True,
                    'model_size_mb': model_size / (1024 * 1024),
                    'training_date': metadata.get('training_date'),
                    'final_loss': metadata.get('metrics', {}).get('best_train_loss'),
                    'training_time': metadata.get('metrics', {}).get('training_time')
                }

            except Exception as e:
                validation_results[equipment_id] = {
                    'valid': False,
                    'error': str(e)
                }

        # Save validation report
        validation_report_path = self.models_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        valid_models = sum(1 for result in validation_results.values() if result.get('valid', False))
        logger.info(f"Model validation completed: {valid_models}/{len(validation_results)} models valid")

        return validation_results

def main():
    """Main training pipeline entry point"""
    parser = argparse.ArgumentParser(description="Telemanom Training Pipeline")
    parser.add_argument('--equipment', type=str, help='Train specific equipment (equipment ID)')
    parser.add_argument('--force', action='store_true', help='Force retrain existing models')
    parser.add_argument('--validate', action='store_true', help='Validate existing models only')
    parser.add_argument('--models-dir', type=str, default='data/models', help='Models directory')

    args = parser.parse_args()

    print("Telemanom Training Pipeline")
    print("=" * 40)

    try:
        pipeline = TelemanamTrainingPipeline(models_dir=args.models_dir)

        if args.validate:
            print("Validating existing models...")
            validation_results = pipeline.validate_models()
            valid_count = sum(1 for r in validation_results.values() if r.get('valid', False))
            print(f"Validation completed: {valid_count}/{len(validation_results)} models valid")

        elif args.equipment:
            print(f"Training model for equipment: {args.equipment}")
            result = pipeline.train_equipment_model(args.equipment, force_retrain=args.force)

            if result['success']:
                print(f"SUCCESS: Model trained in {result['training_time']:.2f} seconds")
            else:
                print(f"FAILED: {result['error']}")

        else:
            print("Training models for all equipment...")
            pipeline.train_all_equipment(force_retrain=args.force)

            stats = pipeline.training_stats
            print(f"Training completed:")
            print(f"  Successful: {stats['trained_models']}")
            print(f"  Failed: {stats['failed_training']}")
            print(f"  Total duration: {(stats['training_end'] - stats['training_start']).total_seconds():.2f} seconds")

        print("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()