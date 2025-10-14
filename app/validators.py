"""Input validation decorators for IoT threat detection API."""
from functools import wraps
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

def validate_prediction_input(endpoint_function):
    """Validate simple 6-field prediction input."""
    @wraps(endpoint_function)
    def validation_wrapper(*args, **kwargs):
        input_data = request.get_json()

        # Check if input data exists
        if not input_data:
            return jsonify({
                'error': 'No input data provided',
                'expected_format': 'JSON object with network traffic fields',
                'status': 'validation_failed'
            }), 400

        # Required network traffic fields for simple prediction
        required_network_fields = [
            'packet_count', 'byte_count', 'duration',
            'syn_flags', 'fin_flags', 'ack_flags'
        ]

        # Check for missing required fields
        missing_fields = [field for field in required_network_fields
                         if field not in input_data]

        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_network_fields,
                'status': 'validation_failed'
            }), 400

        # Define field validation rules
        field_validators = {
            'packet_count': lambda value: isinstance(value, (int, float)) and value >= 0,
            'byte_count': lambda value: isinstance(value, (int, float)) and value >= 0,
            'duration': lambda value: isinstance(value, (int, float)) and value > 0,
            'syn_flags': lambda value: isinstance(value, (int, float)) and value >= 0,
            'fin_flags': lambda value: isinstance(value, (int, float)) and value >= 0,
            'ack_flags': lambda value: isinstance(value, (int, float)) and value >= 0
        }

        # Validate each field
        for field_name, validator_function in field_validators.items():
            field_value = input_data.get(field_name, 0)
            if not validator_function(field_value):
                return jsonify({
                    'error': f'Invalid value for {field_name}: {field_value}',
                    'validation_rule': f'{field_name} must be a non-negative number',
                    'status': 'validation_failed'
                }), 400

        return endpoint_function(*args, **kwargs)

    return validation_wrapper

# Advanced 42-field validation for complete IoT network analysis
REQUIRED_IOT_FEATURES = [
    'flow_duration', 'Duration', 'Rate', 'Srate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

def validate_iot_features(endpoint_function):
    """Validate comprehensive 42-field IoT network features."""
    @wraps(endpoint_function)
    def validation_wrapper(*args, **kwargs):
        try:
            input_data = request.get_json()

            if input_data is None:
                return jsonify({
                    'error': 'Invalid or missing JSON data',
                    'expected_format': 'JSON object with 42 IoT network features',
                    'status': 'validation_failed'
                }), 400

            # Check for missing required features
            missing_features = [feature for feature in REQUIRED_IOT_FEATURES
                              if feature not in input_data]

            if missing_features:
                return jsonify({
                    'error': 'Missing required IoT network features',
                    'missing_features': missing_features,
                    'total_required_features': len(REQUIRED_IOT_FEATURES),
                    'status': 'validation_failed'
                }), 400

            # Log unexpected additional fields
            extra_fields = [field for field in input_data.keys()
                          if field not in REQUIRED_IOT_FEATURES]

            if extra_fields:
                logger.warning(f"Received unexpected fields: {extra_fields}")
            
            validation_errors = []
            
            # Validate each IoT network feature
            for feature_name in REQUIRED_IOT_FEATURES:
                feature_value = input_data[feature_name]

                # Check if value is a valid number
                if not isinstance(feature_value, (int, float)):
                    validation_errors.append(
                        f"{feature_name}: must be numeric, received {type(feature_value).__name__}"
                    )
                    continue

                # Check for invalid numeric values
                if (str(feature_value).lower() in ['inf', '-inf', 'nan'] or
                    feature_value != feature_value):  # NaN check
                    validation_errors.append(f"{feature_name}: invalid numeric value {feature_value}")
                    continue

                # Validate feature-specific ranges
                range_error = validate_feature_range(feature_name, feature_value)
                if range_error:
                    validation_errors.append(range_error)
            
            if validation_errors:
                return jsonify({
                    'error': 'Invalid field values',
                    'validation_errors': validation_errors,
                    'status': 'validation_failed'
                }), 400
            
            logger.info(f"Validation successful for {len(REQUIRED_IOT_FEATURES)} IoT features")

            return endpoint_function(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({
                'error': 'Validation system error',
                'message': str(e),
                'status': 'system_error'
            }), 500
    
    return validation_wrapper

def validate_feature_range(feature_name, feature_value):
    """Validate logical ranges for IoT network traffic features."""

    # Duration and timing related features
    if ('duration' in feature_name.lower() or
        feature_name in ['Duration', 'IAT']):
        if feature_value < 0:
            return f"{feature_name}: duration cannot be negative, received {feature_value}"
        if feature_value > 3600:
            return f"{feature_name}: duration exceeds maximum (3600 seconds), received {feature_value}"

    # Rate and speed features
    elif ('rate' in feature_name.lower() or
          feature_name in ['Rate', 'Srate']):
        if feature_value < 0:
            return f"{feature_name}: rate cannot be negative, received {feature_value}"
        if feature_value > 1000000:
            return f"{feature_name}: rate exceeds maximum (1M/sec), received {feature_value}"

    # Flag and count features
    elif ('flag' in feature_name.lower() or
          'count' in feature_name.lower()):
        if feature_value < 0:
            return f"{feature_name}: count cannot be negative, received {feature_value}"
        if feature_value > 100000:
            return f"{feature_name}: count exceeds maximum (100k), received {feature_value}"

    # Protocol binary flags (must be 0 or 1)
    elif feature_name in ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                         'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']:
        if feature_value not in [0, 1]:
            return f"{feature_name}: protocol flag must be 0 or 1, received {feature_value}"

    # Size and volume features
    elif ('size' in feature_name.lower() or
          feature_name in ['Tot sum', 'Tot size', 'Min', 'Max', 'AVG']):
        if feature_value < 0:
            return f"{feature_name}: size cannot be negative, received {feature_value}"
        if feature_value > 1e12:
            return f"{feature_name}: size exceeds reasonable maximum, received {feature_value}"

    # Statistical measures
    elif feature_name in ['Std', 'Variance']:
        if feature_value < 0:
            return f"{feature_name}: statistical measure cannot be negative, received {feature_value}"

    # Correlation measures
    elif feature_name in ['Covariance']:
        if abs(feature_value) > 1:
            return f"{feature_name}: covariance must be between -1 and 1, received {feature_value}"

    # Count features
    elif feature_name == 'Number':
        if feature_value < 0:
            return f"{feature_name}: count cannot be negative, received {feature_value}"

    # No validation error
    return None
