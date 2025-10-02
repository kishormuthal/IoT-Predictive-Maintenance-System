"""
Centralized Configuration Manager
Handles loading, validation, and access to all system configuration
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""

    field: str
    message: str
    expected_type: Optional[type] = None
    actual_value: Optional[Any] = None


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""

    name: str
    debug: bool
    log_level: str
    database_uri: str
    mlflow_tracking_uri: Optional[str] = None
    redis_enabled: bool = False
    kafka_enabled: bool = False


class ConfigurationManager:
    """
    Centralized configuration management system

    Features:
    - Environment-specific configurations (dev, staging, prod)
    - YAML-based configuration with override support
    - Environment variable support
    - Configuration validation
    - Hot reload capability
    - Type safety and defaults
    """

    _instance = None
    _config: Dict[str, Any] = {}
    _env: str = "development"
    _config_path: Path = None
    _last_loaded: Optional[datetime] = None

    def __new__(cls):
        """Singleton pattern to ensure single configuration instance"""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager"""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._config = {}
            self._env = os.getenv("ENVIRONMENT", "development")
            self._config_path = None
            self._last_loaded = None
            self._validators = {}
            self._setup_validators()

    def _setup_validators(self):
        """Setup configuration validators"""
        self._validators = {
            "system.log_level": lambda v: v in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "dashboard.server.port": lambda v: isinstance(v, int) and 1024 <= v <= 65535,
            "preprocessing.window.size": lambda v: isinstance(v, int) and v > 0,
            "forecasting.general.forecast_horizon": lambda v: isinstance(v, int) and v > 0,
        }

    def load_config(
        self,
        config_path: str = "config/config.yaml",
        env: Optional[str] = None,
        override_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to main configuration file
            env: Environment name (development, staging, production)
            override_path: Optional path to override configuration

        Returns:
            Loaded configuration dictionary
        """
        try:
            # Determine environment
            self._env = env or os.getenv("ENVIRONMENT", "development")

            # Load main configuration
            main_config_path = Path(config_path)
            if not main_config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(main_config_path, "r") as f:
                self._config = yaml.safe_load(f)

            self._config_path = main_config_path

            # Load environment-specific overrides
            env_config_path = main_config_path.parent / f"config.{self._env}.yaml"
            if env_config_path.exists():
                logger.info(f"Loading environment config: {env_config_path}")
                with open(env_config_path, "r") as f:
                    env_config = yaml.safe_load(f)
                    self._merge_configs(self._config, env_config)

            # Load custom overrides
            if override_path and Path(override_path).exists():
                logger.info(f"Loading override config: {override_path}")
                with open(override_path, "r") as f:
                    override_config = yaml.safe_load(f)
                    self._merge_configs(self._config, override_config)

            # Apply environment variables
            self._apply_env_vars()

            # Validate configuration
            self._validate_config()

            # Set environment in config
            self._config["environment"] = self._env
            self._last_loaded = datetime.now()

            logger.info(f"Configuration loaded successfully for environment: {self._env}")
            return self._config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        Recursively merge override config into base config

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
        return base

    def _apply_env_vars(self):
        """Apply environment variable overrides"""
        # Database connection
        if db_uri := os.getenv("DATABASE_URI"):
            self._set_nested("data_ingestion.database.postgresql.uri", db_uri)

        # MLflow tracking URI
        if mlflow_uri := os.getenv("MLFLOW_TRACKING_URI"):
            self._set_nested("mlflow.tracking_uri", mlflow_uri)

        # Redis settings
        if redis_host := os.getenv("REDIS_HOST"):
            self._set_nested("data_ingestion.redis.host", redis_host)

        if redis_port := os.getenv("REDIS_PORT"):
            self._set_nested("data_ingestion.redis.port", int(redis_port))

        # Kafka settings
        if kafka_servers := os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
            self._set_nested("data_ingestion.kafka.bootstrap_servers", kafka_servers)

        # Dashboard settings
        if dashboard_port := os.getenv("DASHBOARD_PORT"):
            self._set_nested("dashboard.server.port", int(dashboard_port))

        # Log level
        if log_level := os.getenv("LOG_LEVEL"):
            self._set_nested("system.log_level", log_level.upper())

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        for path, validator in self._validators.items():
            value = self.get(path)
            if value is not None and not validator(value):
                errors.append(f"Invalid value for {path}: {value}")

        if errors:
            raise ConfigValidationError(
                field="multiple",
                message=f"Configuration validation failed: {'; '.join(errors)}",
            )

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            path: Dot-separated path (e.g., 'dashboard.server.port')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        keys = path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def _set_nested(self, path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def set(self, path: str, value: Any):
        """
        Set configuration value

        Args:
            path: Dot-separated path
            value: Value to set
        """
        self._set_nested(path, value)
        logger.debug(f"Configuration updated: {path} = {value}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section

        Args:
            section: Section name (e.g., 'dashboard', 'mlflow')

        Returns:
            Section configuration dictionary
        """
        return self.get(section, {})

    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file

        Returns:
            Reloaded configuration
        """
        if self._config_path:
            logger.info("Reloading configuration...")
            return self.load_config(str(self._config_path), self._env)
        else:
            logger.warning("No configuration file loaded, cannot reload")
            return self._config

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self._env == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self._env == "development"

    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self._env == "staging"

    @property
    def environment(self) -> str:
        """Get current environment name"""
        return self._env

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self._config.copy()

    def get_env_config(self) -> EnvironmentConfig:
        """Get environment-specific configuration object"""
        return EnvironmentConfig(
            name=self._env,
            debug=self.get("system.debug", False),
            log_level=self.get("system.log_level", "INFO"),
            database_uri=self._get_database_uri(),
            mlflow_tracking_uri=self.get("mlflow.tracking_uri"),
            redis_enabled=self.get("data_ingestion.redis.enabled", False),
            kafka_enabled=self.get("data_ingestion.kafka.enabled", False),
        )

    def _get_database_uri(self) -> str:
        """Construct database URI from configuration"""
        db_type = self.get("data_ingestion.database.type", "sqlite")

        if db_type == "sqlite":
            db_path = self.get("data_ingestion.database.sqlite.path", "./data/iot_telemetry.db")
            return f"sqlite:///{db_path}"

        elif db_type == "postgresql":
            host = self.get("data_ingestion.database.postgresql.host", "localhost")
            port = self.get("data_ingestion.database.postgresql.port", 5432)
            database = self.get("data_ingestion.database.postgresql.database", "iot_telemetry")
            username = self.get("data_ingestion.database.postgresql.username", "iot_user")
            password = self.get("data_ingestion.database.postgresql.password", "")
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        if model_name in ["lstm_predictor", "lstm_autoencoder", "lstm_vae"]:
            return self.get_section(f"anomaly_detection.{model_name}")
        elif model_name in ["transformer", "lstm"]:
            return self.get_section(f"forecasting.{model_name}")
        else:
            return {}

    def get_paths(self) -> Dict[str, Path]:
        """Get all configured paths as Path objects"""
        paths_config = self.get_section("paths")
        return {key: Path(value) for key, value in paths_config.items() if isinstance(value, str)}

    def ensure_paths(self):
        """Ensure all configured directories exist"""
        paths = self.get_paths()
        for name, path in paths.items():
            if not name.endswith("_pattern"):  # Skip pattern keys
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured path exists: {path}")

    def export_config(self, output_path: str, include_defaults: bool = True):
        """
        Export current configuration to YAML file

        Args:
            output_path: Path to save configuration
            include_defaults: Include default values
        """
        with open(output_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration exported to: {output_path}")

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard-specific configuration"""
        return {
            "server": self.get_section("dashboard.server"),
            "ui": self.get_section("dashboard.ui"),
            "components": self.get_section("dashboard.components"),
            "pages": self.get("dashboard.pages", []),
            "performance": self.get_section("dashboard.performance"),
        }

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow-specific configuration"""
        return self.get_section("mlflow")

    def get_alert_config(self) -> Dict[str, Any]:
        """Get alert system configuration"""
        return self.get_section("alerts")

    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigurationManager(env={self._env}, loaded={self._last_loaded})"

    def __str__(self) -> str:
        """String representation"""
        return f"Config({self._env}): {len(self._config)} sections"


# Global configuration instance
_global_config = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """Get global configuration manager instance"""
    return _global_config


def load_config(config_path: str = "config/config.yaml", env: Optional[str] = None) -> ConfigurationManager:
    """
    Load configuration and return manager instance

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        Configuration manager instance
    """
    config_manager = get_config()
    config_manager.load_config(config_path, env)
    return config_manager


# Convenience functions
def get_value(path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation"""
    return get_config().get(path, default)


def get_section(section: str) -> Dict[str, Any]:
    """Get configuration section"""
    return get_config().get_section(section)


def is_production() -> bool:
    """Check if running in production"""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development"""
    return get_config().is_development()
