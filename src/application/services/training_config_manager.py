"""
Training Configuration Manager
Centralized configuration management for training operations
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class TrainingConfigManager:
    """
    Manages training configurations with support for:
    - Equipment-specific overrides
    - Environment-based configurations
    - Configuration validation
    - Dynamic configuration updates
    """

    def __init__(self, config_path: str = "training/config/training_config.yaml"):
        """
        Initialize configuration manager

        Args:
            config_path: Path to main training configuration file
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._base_config = None
        self._environment_configs = {}
        self._equipment_overrides = {}

        # Load configurations
        self._load_configurations()

        logger.info(f"Training configuration manager initialized with {config_path}")

    def _load_configurations(self):
        """Load all configuration files"""
        try:
            # Load base configuration
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self._base_config = yaml.safe_load(f)
                logger.info(f"Loaded base configuration from {self.config_path}")
            else:
                logger.warning(f"Base configuration file not found: {self.config_path}")
                self._base_config = self._get_default_config()

            # Load environment-specific configurations
            env_config_dir = self.config_dir / "environments"
            if env_config_dir.exists():
                for env_file in env_config_dir.glob("*.yaml"):
                    env_name = env_file.stem
                    with open(env_file, "r") as f:
                        self._environment_configs[env_name] = yaml.safe_load(f)
                    logger.info(f"Loaded environment config: {env_name}")

            # Load equipment overrides
            self._equipment_overrides = self._base_config.get("equipment_overrides", {})

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            self._base_config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available"""
        return {
            "global": {
                "project_name": "iot_predictive_maintenance",
                "version": "1.0.0",
                "random_seed": 42,
                "data_root": "./data/raw",
                "model_output_path": "./models",
                "log_level": "INFO",
            },
            "telemanom": {
                "enabled": True,
                "sequence_length": 250,
                "lstm_units": [80, 80],
                "dropout_rate": 0.3,
                "prediction_length": 10,
                "epochs": 35,
                "batch_size": 70,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "min_training_samples": 1000,
                "max_training_samples": 5000,
            },
            "transformer": {
                "enabled": True,
                "sequence_length": 168,
                "forecast_horizon": 24,
                "d_model": 128,
                "num_heads": 8,
                "num_layers": 4,
                "dff": 512,
                "dropout_rate": 0.1,
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "patience": 15,
                "min_training_samples": 2000,
                "max_training_samples": 10000,
            },
            "training": {
                "parallel_training": False,
                "max_workers": 4,
                "save_best_models": True,
                "save_checkpoints": True,
            },
            "equipment_overrides": {},
        }

    def get_config(self, environment: str = None) -> Dict[str, Any]:
        """
        Get complete configuration for specified environment

        Args:
            environment: Environment name (development, production, etc.)

        Returns:
            Complete configuration dictionary
        """
        try:
            # Start with base configuration
            config = self._base_config.copy()

            # Apply environment-specific overrides
            if environment and environment in self._environment_configs:
                env_config = self._environment_configs[environment]
                config = self._merge_configs(config, env_config)
                logger.info(f"Applied environment configuration: {environment}")

            # Apply any runtime environment variables
            config = self._apply_environment_variables(config)

            return config

        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return self._get_default_config()

    def get_sensor_config(
        self, sensor_id: str, model_type: str, environment: str = None
    ) -> Dict[str, Any]:
        """
        Get configuration for specific sensor and model type

        Args:
            sensor_id: Equipment sensor ID
            model_type: Model type ('telemanom' or 'transformer')
            environment: Environment name

        Returns:
            Sensor-specific configuration
        """
        try:
            # Get base configuration for model type
            base_config = self.get_config(environment)
            model_config = base_config.get(model_type, {}).copy()

            # Apply equipment-specific overrides
            equipment_overrides = self._equipment_overrides.get(sensor_id, {})
            model_overrides = equipment_overrides.get(model_type, {})

            if model_overrides:
                logger.info(
                    f"Applying overrides for {sensor_id} {model_type}: {model_overrides}"
                )
                model_config = self._merge_configs(model_config, model_overrides)

            # Add sensor identification
            model_config["sensor_id"] = sensor_id
            model_config["model_type"] = model_type

            return model_config

        except Exception as e:
            logger.error(f"Error getting sensor configuration for {sensor_id}: {e}")
            return self._get_default_config().get(model_type, {})

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        try:
            # Map of environment variables to config paths
            env_mappings = {
                "IOT_DATA_ROOT": ["global", "data_root"],
                "IOT_MODEL_PATH": ["global", "model_output_path"],
                "IOT_LOG_LEVEL": ["global", "log_level"],
                "IOT_PARALLEL_TRAINING": ["training", "parallel_training"],
                "IOT_MAX_WORKERS": ["training", "max_workers"],
                "IOT_TELEMANOM_EPOCHS": ["telemanom", "epochs"],
                "IOT_TRANSFORMER_EPOCHS": ["transformer", "epochs"],
            }

            for env_var, config_path in env_mappings.items():
                if env_var in os.environ:
                    value = os.environ[env_var]

                    # Convert to appropriate type
                    if config_path[-1] in ["parallel_training"]:
                        value = value.lower() in ("true", "1", "yes")
                    elif config_path[-1] in ["max_workers", "epochs"]:
                        value = int(value)

                    # Set in config
                    current = config
                    for key in config_path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[config_path[-1]] = value

                    logger.info(f"Applied environment override: {env_var} = {value}")

            return config

        except Exception as e:
            logger.error(f"Error applying environment variables: {e}")
            return config

    def validate_config(
        self, config: Dict[str, Any], model_type: str = None
    ) -> Dict[str, Any]:
        """
        Validate configuration parameters

        Args:
            config: Configuration to validate
            model_type: Specific model type to validate

        Returns:
            Validation results
        """
        validation_results = {"valid": True, "errors": [], "warnings": []}

        try:
            # Global validation
            if "global" in config:
                global_config = config["global"]

                # Check required paths
                data_root = Path(global_config.get("data_root", "./data/raw"))
                if not data_root.exists():
                    validation_results["warnings"].append(
                        f"Data root directory does not exist: {data_root}"
                    )

                model_path = Path(global_config.get("model_output_path", "./models"))
                if not model_path.parent.exists():
                    validation_results["errors"].append(
                        f"Model output parent directory does not exist: {model_path.parent}"
                    )

            # Model-specific validation
            models_to_validate = (
                [model_type] if model_type else ["telemanom", "transformer"]
            )

            for model in models_to_validate:
                if model in config:
                    model_config = config[model]

                    # Telemanom validation
                    if model == "telemanom":
                        if model_config.get("sequence_length", 0) < 10:
                            validation_results["errors"].append(
                                "Telemanom sequence_length must be at least 10"
                            )

                        if model_config.get("epochs", 0) < 1:
                            validation_results["errors"].append(
                                "Telemanom epochs must be at least 1"
                            )

                        if model_config.get("batch_size", 0) < 1:
                            validation_results["errors"].append(
                                "Telemanom batch_size must be at least 1"
                            )

                    # Transformer validation
                    elif model == "transformer":
                        if model_config.get("sequence_length", 0) < 24:
                            validation_results["warnings"].append(
                                "Transformer sequence_length should be at least 24 for daily patterns"
                            )

                        if (
                            model_config.get("d_model", 0)
                            % model_config.get("num_heads", 1)
                            != 0
                        ):
                            validation_results["errors"].append(
                                "Transformer d_model must be divisible by num_heads"
                            )

                        if model_config.get("forecast_horizon", 0) < 1:
                            validation_results["errors"].append(
                                "Transformer forecast_horizon must be at least 1"
                            )

            # Training configuration validation
            if "training" in config:
                training_config = config["training"]

                max_workers = training_config.get("max_workers", 1)
                if max_workers < 1 or max_workers > 16:
                    validation_results["warnings"].append(
                        "max_workers should be between 1 and 16"
                    )

            validation_results["valid"] = len(validation_results["errors"]) == 0

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")

        return validation_results

    def update_equipment_override(
        self, sensor_id: str, model_type: str, overrides: Dict[str, Any]
    ):
        """
        Update equipment-specific configuration overrides

        Args:
            sensor_id: Equipment sensor ID
            model_type: Model type ('telemanom' or 'transformer')
            overrides: Configuration overrides to apply
        """
        try:
            if sensor_id not in self._equipment_overrides:
                self._equipment_overrides[sensor_id] = {}

            if model_type not in self._equipment_overrides[sensor_id]:
                self._equipment_overrides[sensor_id][model_type] = {}

            # Merge overrides
            self._equipment_overrides[sensor_id][model_type].update(overrides)

            # Update base config
            if "equipment_overrides" not in self._base_config:
                self._base_config["equipment_overrides"] = {}

            self._base_config["equipment_overrides"] = self._equipment_overrides

            logger.info(f"Updated equipment override for {sensor_id} {model_type}")

        except Exception as e:
            logger.error(f"Error updating equipment override: {e}")

    def save_config(self, config: Dict[str, Any] = None, file_path: str = None):
        """
        Save configuration to file

        Args:
            config: Configuration to save (uses current base config if None)
            file_path: File path to save to (uses default if None)
        """
        try:
            config_to_save = config or self._base_config
            save_path = Path(file_path) if file_path else self.config_path

            # Create backup of existing config
            if save_path.exists():
                backup_path = save_path.with_suffix(
                    f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
                )
                save_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Save new config
            with open(save_path, "w") as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def create_environment_config(
        self, environment: str, config_overrides: Dict[str, Any]
    ):
        """
        Create environment-specific configuration

        Args:
            environment: Environment name
            config_overrides: Configuration overrides for this environment
        """
        try:
            env_config_dir = self.config_dir / "environments"
            env_config_dir.mkdir(exist_ok=True)

            env_config_path = env_config_dir / f"{environment}.yaml"

            with open(env_config_path, "w") as f:
                yaml.dump(config_overrides, f, default_flow_style=False, indent=2)

            # Update in-memory cache
            self._environment_configs[environment] = config_overrides

            logger.info(f"Created environment configuration: {environment}")

        except Exception as e:
            logger.error(f"Error creating environment configuration: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all available configurations"""
        try:
            return {
                "base_config_path": str(self.config_path),
                "base_config_exists": self.config_path.exists(),
                "environment_configs": list(self._environment_configs.keys()),
                "equipment_overrides_count": len(self._equipment_overrides),
                "equipment_with_overrides": list(self._equipment_overrides.keys()),
                "model_types_configured": (
                    [
                        key
                        for key in self._base_config.keys()
                        if key in ["telemanom", "transformer"]
                    ]
                    if self._base_config
                    else []
                ),
            }

        except Exception as e:
            logger.error(f"Error getting configuration summary: {e}")
            return {"error": str(e)}
