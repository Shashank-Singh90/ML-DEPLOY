#!/usr/bin/env python3
"""
Fix Monitoring Stats Endpoint
This script identifies and fixes import issues in the main.py monitoring endpoint
"""

import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_main_py_imports():
    """Fix import issues in main.py"""
    
    main_py_path = 'app/main.py'
    
    if not os.path.exists(main_py_path):
        logger.error(f"❌ {main_py_path} not found")
        return False
    
    logger.info(f"🔧 Fixing imports in {main_py_path}")
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Fix the datetime import issue
    if 'datetime.timedelta' in content:
        content = content.replace('datetime.timedelta', 'timedelta')
        logger.info("✅ Fixed datetime.timedelta import")
    
    # Ensure timedelta is imported
    if 'from datetime import datetime' in content and 'timedelta' not in content.split('from datetime import')[1].split('\n')[0]:
        content = content.replace(
            'from datetime import datetime',
            'from datetime import datetime, timedelta'
        )
        logger.info("✅ Added timedelta to datetime imports")
    
    # Write back the fixed content
    with open(main_py_path, 'w') as f:
        f.write(content)
    
    logger.info("✅ Fixed main.py imports")
    return True

def create_simple_monitoring_stats():
    """Create a simplified monitoring stats function that doesn't have import issues"""
    
    monitoring_fix = '''
def get_monitoring_stats_safe():
    """Get monitoring statistics without datetime import issues"""
    try:
        from datetime import datetime, timedelta
        
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Calculate recent statistics  
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_hour_predictions = [
            p for p in recent_predictions 
            if p['timestamp'] > one_hour_ago
        ]
        
        attack_count_hour = sum(1 for p in recent_hour_predictions if p['prediction'] == 1)
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'total_predictions': len(recent_predictions),
            'recent_hour': {
                'total_predictions': len(recent_hour_predictions),
                'attack_predictions': attack_count_hour,
                'attack_rate': attack_count_hour / max(len(recent_hour_predictions), 1)
            },
            'current_metrics': {
                'avg_threat_score': sum(recent_threat_scores) / len(recent_threat_scores) if recent_threat_scores else 0,
                'avg_confidence': sum(p['confidence'] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
            },
            'system_health': 'healthy' if len(recent_predictions) > 0 else 'no_traffic'
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Monitoring stats error: {str(e)}")
        return {
            'error': f'Failed to retrieve monitoring stats: {str(e)}',
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }
'''
    
    # Write the fix to a separate file
    with open('monitoring_stats_fix.py', 'w') as f:
        f.write(monitoring_fix)
    
    logger.info("✅ Created monitoring stats fix")
    return True

def apply_monitoring_fix():
    """Apply the monitoring fix to main.py"""
    
    main_py_path = 'app/main.py'
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic monitoring/stats endpoint
    old_endpoint = '''@app.route('/monitoring/stats', methods=['GET'])
def monitoring_stats():
    """Get current monitoring statistics"""
    try:
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Calculate recent statistics
        one_hour_ago = datetime.now() - datetime.timedelta(hours=1)
        recent_hour_predictions = [
            p for p in recent_predictions 
            if p['timestamp'] > one_hour_ago
        ]
        
        attack_count_hour = sum(1 for p in recent_hour_predictions if p['prediction'] == 1)
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'total_predictions': len(recent_predictions),
            'recent_hour': {
                'total_predictions': len(recent_hour_predictions),
                'attack_predictions': attack_count_hour,
                'attack_rate': attack_count_hour / max(len(recent_hour_predictions), 1)
            },
            'current_metrics': {
                'avg_threat_score': sum(recent_threat_scores) / len(recent_threat_scores) if recent_threat_scores else 0,
                'avg_confidence': sum(p['confidence'] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
            },
            'system_health': 'healthy' if len(recent_predictions) > 0 else 'no_traffic'
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Monitoring stats error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve monitoring stats",
            "message": str(e)
        }), 500'''
    
    new_endpoint = '''@app.route('/monitoring/stats', methods=['GET'])
def monitoring_stats():
    """Get current monitoring statistics"""
    try:
        from datetime import timedelta
        
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Calculate recent statistics
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_hour_predictions = [
            p for p in recent_predictions 
            if p['timestamp'] > one_hour_ago
        ]
        
        attack_count_hour = sum(1 for p in recent_hour_predictions if p['prediction'] == 1)
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'total_predictions': len(recent_predictions),
            'recent_hour': {
                'total_predictions': len(recent_hour_predictions),
                'attack_predictions': attack_count_hour,
                'attack_rate': attack_count_hour / max(len(recent_hour_predictions), 1)
            },
            'current_metrics': {
                'avg_threat_score': sum(recent_threat_scores) / len(recent_threat_scores) if recent_threat_scores else 0,
                'avg_confidence': sum(p['confidence'] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
            },
            'system_health': 'healthy' if len(recent_predictions) > 0 else 'no_traffic'
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Monitoring stats error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve monitoring stats",
            "message": str(e)
        }), 500'''
    
    # Replace the endpoint
    if old_endpoint in content:
        content = content.replace(old_endpoint, new_endpoint)
        logger.info("✅ Fixed monitoring stats endpoint")
    else:
        logger.warning("⚠️  Could not find exact monitoring stats endpoint to replace")
    
    # Write back the content
    with open(main_py_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Main execution function"""
    print("🔧 Fixing Monitoring Endpoint Issues...")
    print("=" * 50)
    
    # Step 1: Fix imports
    if not fix_main_py_imports():
        print("❌ Failed to fix imports")
        return False
    
    # Step 2: Create monitoring fix
    if not create_simple_monitoring_stats():
        print("❌ Failed to create monitoring fix")
        return False
    
    # Step 3: Apply the fix
    if not apply_monitoring_fix():
        print("❌ Failed to apply monitoring fix")
        return False
    
    print("=" * 50)
    print("✅ MONITORING ENDPOINT FIX COMPLETE!")
    print("🚀 /monitoring/stats endpoint should now work properly")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
