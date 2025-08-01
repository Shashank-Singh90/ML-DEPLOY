#!/usr/bin/env python3
"""
Quick fix for the monitoring endpoint encoding issue
"""

import os

def fix_main_py_encoding():
    """Fix encoding issues in main.py"""
    main_py_path = 'app/main.py'
    
    try:
        # Read with utf-8 encoding to handle special characters
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the datetime import issue
        if 'datetime.timedelta' in content:
            content = content.replace('datetime.timedelta', 'timedelta')
            print("✅ Fixed datetime.timedelta import")
        
        # Ensure timedelta is imported
        if ('from datetime import datetime' in content and 
            'timedelta' not in content.split('from datetime import')[1].split('\n')[0]):
            content = content.replace(
                'from datetime import datetime',
                'from datetime import datetime, timedelta'
            )
            print("✅ Added timedelta to datetime imports")
        
        # Write back with utf-8 encoding
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("🎉 Successfully fixed encoding issues in main.py!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing main.py: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing monitoring endpoint encoding issues...")
    success = fix_main_py_encoding()
    if success:
        print("✅ Fix complete! Restart your API to apply changes.")
    else:
        print("❌ Fix failed. The API should still work for predictions.")
