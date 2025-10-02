                            recommendations['sensors_needing_retraining'].append({
                                'sensor_id': sensor_id,
                                'equipment_type': equipment.equipment_type.value,
                                'criticality': equipment.criticality.value,
                                'performance_score': metadata.performance_score,
                                'reason': 'Low performance score'
                            })
                        else:
                            recommendations['well_performing_sensors'].append({
                                'sensor_id': sensor_id,
                                'equipment_type': equipment.equipment_type.value,
                                'performance_score': metadata.performance_score
                            })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating training recommendations: {e}")
            return {
                'error': str(e),
                'sensors_needing_training': [],
                'sensors_needing_retraining': [],
                'well_performing_sensors': []
            }
