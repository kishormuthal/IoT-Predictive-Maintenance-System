"""
Equipment Configuration for 12-Equipment IoT System
Maps NASA SMAP/MSL data to 12 equipment units with primary sensors
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class CriticalityLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EquipmentType(Enum):
    POWER = "POWER"
    COMMUNICATION = "COMMUNICATION"
    ATTITUDE = "ATTITUDE"
    THERMAL = "THERMAL"
    PAYLOAD = "PAYLOAD"
    MOBILITY = "MOBILITY"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    SCIENTIFIC = "SCIENTIFIC"
    NAVIGATION = "NAVIGATION"


@dataclass
class EquipmentConfig:
    """Configuration for a single equipment unit"""

    equipment_id: str
    name: str
    equipment_type: EquipmentType
    data_source: str  # 'smap' or 'msl'
    channel_index: int
    sensor_type: str
    unit: str
    criticality: CriticalityLevel
    location: str
    description: str
    normal_range: tuple
    warning_threshold: float
    critical_threshold: float


# 12 Equipment Configuration: 6 SMAP + 6 MSL
EQUIPMENT_REGISTRY: Dict[str, EquipmentConfig] = {
    # SMAP Equipment (6 units)
    "SMAP-PWR-001": EquipmentConfig(
        equipment_id="SMAP-PWR-001",
        name="Primary Power System",
        equipment_type=EquipmentType.POWER,
        data_source="smap",
        channel_index=0,
        sensor_type="Voltage",
        unit="V",
        criticality=CriticalityLevel.CRITICAL,
        location="SMAP Satellite",
        description="Main power bus voltage monitoring",
        normal_range=(24.0, 28.0),
        warning_threshold=23.0,
        critical_threshold=22.0,
    ),
    "SMAP-COM-001": EquipmentConfig(
        equipment_id="SMAP-COM-001",
        name="Communication System",
        equipment_type=EquipmentType.COMMUNICATION,
        data_source="smap",
        channel_index=1,
        sensor_type="Signal_Strength",
        unit="dBm",
        criticality=CriticalityLevel.HIGH,
        location="SMAP Satellite",
        description="Communication signal strength monitoring",
        normal_range=(-80.0, -20.0),
        warning_threshold=-85.0,
        critical_threshold=-90.0,
    ),
    "SMAP-ATT-001": EquipmentConfig(
        equipment_id="SMAP-ATT-001",
        name="Attitude Control System",
        equipment_type=EquipmentType.ATTITUDE,
        data_source="smap",
        channel_index=2,
        sensor_type="Angular_Velocity",
        unit="deg/s",
        criticality=CriticalityLevel.CRITICAL,
        location="SMAP Satellite",
        description="Attitude control gyroscope monitoring",
        normal_range=(-0.5, 0.5),
        warning_threshold=0.8,
        critical_threshold=1.0,
    ),
    "SMAP-THM-001": EquipmentConfig(
        equipment_id="SMAP-THM-001",
        name="Thermal Management System",
        equipment_type=EquipmentType.THERMAL,
        data_source="smap",
        channel_index=3,
        sensor_type="Temperature",
        unit="°C",
        criticality=CriticalityLevel.HIGH,
        location="SMAP Satellite",
        description="Thermal system temperature monitoring",
        normal_range=(-10.0, 50.0),
        warning_threshold=55.0,
        critical_threshold=60.0,
    ),
    "SMAP-PAY-001": EquipmentConfig(
        equipment_id="SMAP-PAY-001",
        name="Payload System",
        equipment_type=EquipmentType.PAYLOAD,
        data_source="smap",
        channel_index=4,
        sensor_type="Performance_Index",
        unit="index",
        criticality=CriticalityLevel.HIGH,
        location="SMAP Satellite",
        description="Payload performance monitoring",
        normal_range=(0.8, 1.0),
        warning_threshold=0.7,
        critical_threshold=0.6,
    ),
    "SMAP-SYS-001": EquipmentConfig(
        equipment_id="SMAP-SYS-001",
        name="System Monitor",
        equipment_type=EquipmentType.PAYLOAD,
        data_source="smap",
        channel_index=5,
        sensor_type="System_Health",
        unit="index",
        criticality=CriticalityLevel.MEDIUM,
        location="SMAP Satellite",
        description="Overall system health monitoring",
        normal_range=(0.9, 1.0),
        warning_threshold=0.8,
        critical_threshold=0.7,
    ),
    # MSL Equipment (6 units)
    "MSL-MOB-001": EquipmentConfig(
        equipment_id="MSL-MOB-001",
        name="Mobility System Primary",
        equipment_type=EquipmentType.MOBILITY,
        data_source="msl",
        channel_index=0,
        sensor_type="Motor_Current",
        unit="A",
        criticality=CriticalityLevel.CRITICAL,
        location="Mars Surface",
        description="Primary mobility motor current monitoring",
        normal_range=(2.0, 8.0),
        warning_threshold=9.0,
        critical_threshold=10.0,
    ),
    "MSL-PWR-001": EquipmentConfig(
        equipment_id="MSL-PWR-001",
        name="Power Management Unit",
        equipment_type=EquipmentType.POWER,
        data_source="msl",
        channel_index=1,
        sensor_type="Battery_Voltage",
        unit="V",
        criticality=CriticalityLevel.CRITICAL,
        location="Mars Surface",
        description="Battery system voltage monitoring",
        normal_range=(26.0, 32.0),
        warning_threshold=25.0,
        critical_threshold=24.0,
    ),
    "MSL-ENV-001": EquipmentConfig(
        equipment_id="MSL-ENV-001",
        name="Environmental Monitoring",
        equipment_type=EquipmentType.ENVIRONMENTAL,
        data_source="msl",
        channel_index=2,
        sensor_type="Ambient_Temperature",
        unit="°C",
        criticality=CriticalityLevel.MEDIUM,
        location="Mars Surface",
        description="Environmental temperature monitoring",
        normal_range=(-80.0, 0.0),
        warning_threshold=-90.0,
        critical_threshold=-100.0,
    ),
    "MSL-SCI-001": EquipmentConfig(
        equipment_id="MSL-SCI-001",
        name="Scientific Instrument Package",
        equipment_type=EquipmentType.SCIENTIFIC,
        data_source="msl",
        channel_index=3,
        sensor_type="Instrument_Power",
        unit="W",
        criticality=CriticalityLevel.HIGH,
        location="Mars Surface",
        description="Scientific instrument power monitoring",
        normal_range=(10.0, 50.0),
        warning_threshold=8.0,
        critical_threshold=5.0,
    ),
    "MSL-COM-001": EquipmentConfig(
        equipment_id="MSL-COM-001",
        name="Communication Array",
        equipment_type=EquipmentType.COMMUNICATION,
        data_source="msl",
        channel_index=4,
        sensor_type="RF_Power",
        unit="W",
        criticality=CriticalityLevel.HIGH,
        location="Mars Surface",
        description="Communication RF power monitoring",
        normal_range=(20.0, 100.0),
        warning_threshold=15.0,
        critical_threshold=10.0,
    ),
    "MSL-NAV-001": EquipmentConfig(
        equipment_id="MSL-NAV-001",
        name="Navigation System",
        equipment_type=EquipmentType.NAVIGATION,
        data_source="msl",
        channel_index=5,
        sensor_type="Position_Accuracy",
        unit="m",
        criticality=CriticalityLevel.CRITICAL,
        location="Mars Surface",
        description="Navigation position accuracy monitoring",
        normal_range=(0.1, 2.0),
        warning_threshold=3.0,
        critical_threshold=5.0,
    ),
}


def get_equipment_by_id(equipment_id: str) -> EquipmentConfig:
    """Get equipment configuration by ID"""
    return EQUIPMENT_REGISTRY.get(equipment_id)


def get_equipment_list() -> List[EquipmentConfig]:
    """Get list of all equipment configurations"""
    return list(EQUIPMENT_REGISTRY.values())


def get_smap_equipment() -> List[EquipmentConfig]:
    """Get SMAP equipment only"""
    return [eq for eq in EQUIPMENT_REGISTRY.values() if eq.data_source == "smap"]


def get_msl_equipment() -> List[EquipmentConfig]:
    """Get MSL equipment only"""
    return [eq for eq in EQUIPMENT_REGISTRY.values() if eq.data_source == "msl"]


def get_critical_equipment() -> List[EquipmentConfig]:
    """Get critical equipment only"""
    return [eq for eq in EQUIPMENT_REGISTRY.values() if eq.criticality == CriticalityLevel.CRITICAL]


def get_equipment_by_type(equipment_type: EquipmentType) -> List[EquipmentConfig]:
    """Get equipment by type"""
    return [eq for eq in EQUIPMENT_REGISTRY.values() if eq.equipment_type == equipment_type]


# Equipment summary statistics
EQUIPMENT_STATS = {
    "total_equipment": len(EQUIPMENT_REGISTRY),
    "smap_equipment": len(get_smap_equipment()),
    "msl_equipment": len(get_msl_equipment()),
    "critical_equipment": len(get_critical_equipment()),
    "equipment_types": len(set(eq.equipment_type for eq in EQUIPMENT_REGISTRY.values())),
    "data_sources": ["smap", "msl"],
}
