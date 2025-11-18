"""
Generate synthetic IoT network traffic data for testing.
Tries to mimic the RT-IoT2022 dataset structure.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_iot_data(n_samples=10000, output_path='data/raw/synthetic_iot_data.csv'):
    """Generate synthetic IoT network traffic data"""

    # Keep it reproducible
    np.random.seed(42)

    print(f"🚀 Generating {n_samples} synthetic IoT samples...")

    # Using distributions that roughly match real IoT traffic patterns
    data = {}

    # Basic network flow stuff
    data['flow_duration'] = np.random.exponential(2.0, n_samples)
    data['Header_Length'] = np.random.choice([20, 40, 60], n_samples, p=[0.7, 0.2, 0.1])
    data['Protocol Type'] = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.6, 0.3, 0.1])
    data['Duration'] = data['flow_duration'] + np.random.normal(0, 0.5, n_samples)
    data['Rate'] = np.random.gamma(2, 50, n_samples)
    data['Srate'] = data['Rate'] * np.random.uniform(0.5, 1.5, n_samples)  # scaled rate
    
    # TCP flag counts - using Poisson since they're counts
    data['fin_flag_number'] = np.random.poisson(0.5, n_samples)
    data['syn_flag_number'] = np.random.poisson(1.0, n_samples)
    data['rst_flag_number'] = np.random.poisson(0.2, n_samples)
    data['psh_flag_number'] = np.random.poisson(2.0, n_samples)
    data['ack_flag_number'] = np.random.poisson(5.0, n_samples)  # ACK is most common
    data['ece_flag_number'] = np.random.poisson(0.1, n_samples)
    data['cwr_flag_number'] = np.random.poisson(0.1, n_samples)
    
    # Packet counts
    data['ack_count'] = data['ack_flag_number'] + np.random.poisson(3, n_samples)
    data['syn_count'] = data['syn_flag_number'] + np.random.poisson(1, n_samples)
    data['fin_count'] = data['fin_flag_number'] + np.random.poisson(0.5, n_samples)
    data['rst_count'] = data['rst_flag_number'] + np.random.poisson(0.2, n_samples)
    
    # Protocol indicators - binary flags for which protocols are used
    data['HTTP'] = np.random.binomial(1, 0.3, n_samples)
    data['HTTPS'] = np.random.binomial(1, 0.2, n_samples)
    data['DNS'] = np.random.binomial(1, 0.1, n_samples)
    data['Telnet'] = np.random.binomial(1, 0.05, n_samples)
    data['SMTP'] = np.random.binomial(1, 0.05, n_samples)
    data['SSH'] = np.random.binomial(1, 0.1, n_samples)
    data['IRC'] = np.random.binomial(1, 0.02, n_samples)  # IRC is pretty rare these days
    data['TCP'] = (data['Protocol Type'] == 'TCP').astype(int)
    data['UDP'] = (data['Protocol Type'] == 'UDP').astype(int)
    data['DHCP'] = np.random.binomial(1, 0.05, n_samples)
    data['ARP'] = np.random.binomial(1, 0.08, n_samples)
    data['ICMP'] = (data['Protocol Type'] == 'ICMP').astype(int)
    data['IPv'] = np.ones(n_samples, dtype=int)  # pretty much everything is IP
    data['LLC'] = np.random.binomial(1, 0.02, n_samples)
    
    # Packet size statistics
    data['Tot sum'] = np.random.gamma(3, 500, n_samples)
    data['Min'] = np.random.gamma(1, 50, n_samples)
    data['Max'] = data['Tot sum'] * np.random.uniform(0.8, 2.0, n_samples)
    data['AVG'] = (data['Tot sum'] + data['Min'] + data['Max']) / 3
    data['Std'] = np.abs(np.random.normal(0, data['AVG'] * 0.3, n_samples))
    data['Tot size'] = data['Tot sum'] * np.random.uniform(0.9, 1.1, n_samples)
    
    # Time-based features
    data['IAT'] = np.random.exponential(0.1, n_samples)  # Inter-arrival time
    data['Number'] = np.random.poisson(10, n_samples)  # Number of packets
    
    # Statistical features - TODO: not sure if these are all meaningful
    data['Magnitue'] = np.random.gamma(2, 100, n_samples)  # keeping the typo to match model
    data['Radius'] = np.random.gamma(1.5, 50, n_samples)
    data['Covariance'] = np.random.uniform(-1, 1, n_samples)
    data['Variance'] = np.random.gamma(1, 100, n_samples)
    data['Weight'] = np.random.gamma(2, 0.5, n_samples)

    # Generate labels - 0 for normal, 1-5 for different attack types
    attack_prob = 0.15  # 15% of traffic is malicious
    is_attack = np.random.binomial(1, attack_prob, n_samples)
    
    # Assign attack types (1-5) for attacks, 0 for normal
    data['label'] = np.where(is_attack,
                           np.random.choice([1, 2, 3, 4, 5], n_samples,
                                         p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                           0)

    # Build the dataframe
    df = pd.DataFrame(data)

    # Make attacks look different from normal traffic
    attack_mask = df['label'] > 0

    if attack_mask.sum() > 0:
        # Attacks usually have higher packet rates
        df.loc[attack_mask, 'Rate'] *= np.random.uniform(2, 5, attack_mask.sum())
        df.loc[attack_mask, 'fin_flag_number'] *= np.random.uniform(0.1, 3, attack_mask.sum())
        df.loc[attack_mask, 'rst_flag_number'] *= np.random.uniform(2, 8, attack_mask.sum())  # more resets
        df.loc[attack_mask, 'Tot size'] *= np.random.uniform(0.1, 0.5, attack_mask.sum())

    # Make sure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write it out
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {n_samples} synthetic IoT samples")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Attack rate: {(df['label'] > 0).mean():.2%}")
    print(f"📏 Shape: {df.shape}")
    print(f"🏷️  Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    print("🔄 Starting IoT data generation...")
    df = generate_synthetic_iot_data(n_samples=50000)  # generate a decent amount

    # Also make a small test set for quick testing
    test_df = df.sample(n=1000, random_state=42)
    test_df.to_csv('data/raw/test_sample.csv', index=False)
    print("✅ Also created small test sample (1000 rows)")
    print("📂 Files created:")
    print("   - data/raw/synthetic_iot_data.csv (50,000 rows)")
    print("   - data/raw/test_sample.csv (1,000 rows)")
    