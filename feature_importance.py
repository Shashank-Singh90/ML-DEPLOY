import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create sample data to understand features
def create_sample_data():
    """Generate example network traffic data"""
    
    # Normal traffic patterns
    normal_traffic = []
    for i in range(100):
        normal_traffic.append({
            'packet_ratio': np.random.uniform(0.8, 1.2),    # Balanced
            'syn_ratio': np.random.uniform(0.01, 0.05),     # Low SYN
            'packet_size': np.random.uniform(500, 1500),    # Regular size
            'packet_rate': np.random.uniform(10, 100),      # Moderate rate
            'label': 0  # 0 = normal
        })
    
    # Attack traffic patterns
    attack_traffic = []
    for i in range(100):
        attack_type = np.random.choice(['syn_flood', 'port_scan', 'ddos'])
        
        if attack_type == 'syn_flood':
            attack_traffic.append({
                'packet_ratio': np.random.uniform(5, 10),    # Sending way more
                'syn_ratio': np.random.uniform(0.6, 0.9),    # Lots of SYN
                'packet_size': np.random.uniform(40, 60),    # Small packets
                'packet_rate': np.random.uniform(500, 1000), # High rate
                'label': 1  # 1 = attack
            })
        elif attack_type == 'port_scan':
            attack_traffic.append({
                'packet_ratio': np.random.uniform(3, 5),
                'syn_ratio': np.random.uniform(0.3, 0.5),
                'packet_size': np.random.uniform(20, 40),    # Tiny packets
                'packet_rate': np.random.uniform(200, 400),
                'label': 1
            })
        else:  # ddos
            attack_traffic.append({
                'packet_ratio': np.random.uniform(0.1, 0.3), # Receiving more
                'syn_ratio': np.random.uniform(0.1, 0.2),
                'packet_size': np.random.uniform(1400, 1500),
                'packet_rate': np.random.uniform(1000, 5000), # Extreme rate
                'label': 1
            })
    
    # Combine and create DataFrame
    all_traffic = normal_traffic + attack_traffic
    df = pd.DataFrame(all_traffic)
    
    return df

# Train a simple model to see feature importance
df = create_sample_data()
print("Dataset created with {} samples".format(len(df)))
print("\nFirst few rows:")
print(df.head())

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Show feature importance
print("\nFeature Importance (what the model pays attention to):")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
    if importance > 0.3:
        print(f"    → This is VERY important for detection!")
    elif importance > 0.2:
        print(f"    → This is moderately important")
    else:
        print(f"    → This helps but isn't critical")
