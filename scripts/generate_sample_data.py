"""
Generate synthetic IoT network traffic data for testing
This mimics the RT-IoT2022 dataset structure
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_iot_data(n_samples=10000, output_path='data/raw/synthetic_iot_data.csv'):
    """Generate synthetic IoT network traffic data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print(f"🚀 Generating {n_samples} synthetic IoT samples...")
    
    # Create synthetic data with realistic distributions
    data = {}
    
    # Basic flow features
    data['flow_duration'] = np.random.exponential(2.0, n_samples)
    data['Header_Length'] = np.random.choice([20, 40, 60], n_samples, p=[0.7, 0.2, 0.1])
    data['Protocol Type'] = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.6, 0.3, 0.1])
    data['Duration'] = data['flow_duration'] + np.random.normal(0, 0.5, n_samples)
    data['Rate'] = np.random.gamma(2, 50, n_samples)
    data['Srate'] = data['Rate'] * np.random.uniform(0.5, 1.5, n_samples)
    
    # TCP Flags
    data['fin_flag_number'] = np.random.poisson(0.5, n_samples)
    data['syn_flag_number'] = np.random.poisson(1.0, n_samples)  
    data['rst_flag_number'] = np.random.poisson(0.2, n_samples)
    data['psh_flag_number'] = np.random.poisson(2.0, n_samples)
    data['ack_flag_number'] = np.random.poisson(5.0, n_samples)
    data['ece_flag_number'] = np.random.poisson(0.1, n_samples)
    data['cwr_flag_number'] = np.random.poisson(0.1, n_samples)
    
    # Packet counts
    data['ack_count'] = data['ack_flag_number'] + np.random.poisson(3, n_samples)
    data['syn_count'] = data['syn_flag_number'] + np.random.poisson(1, n_samples)
    data['fin_count'] = data['fin_flag_number'] + np.random.poisson(0.5, n_samples)
    data['rst_count'] = data['rst_flag_number'] + np.random.poisson(0.2, n_samples)
    
    # Protocol indicators (binary)
    data['HTTP'] = np.random.binomial(1, 0.3, n_samples)
    data['HTTPS'] = np.random.binomial(1, 0.2, n_samples)  
    data['DNS'] = np.random.binomial(1, 0.1, n_samples)
    data['Telnet'] = np.random.binomial(1, 0.05, n_samples)
    data['SMTP'] = np.random.binomial(1, 0.05, n_samples)
    data['SSH'] = np.random.binomial(1, 0.1, n_samples)
    data['IRC'] = np.random.binomial(1, 0.02, n_samples)
    data['TCP'] = (data['Protocol Type'] == 'TCP').astype(int)
    data['UDP'] = (data['Protocol Type'] == 'UDP').astype(int)
    data['DHCP'] = np.random.binomial(1, 0.05, n_samples)
    data['ARP'] = np.random.binomial(1, 0.08, n_samples)
    data['ICMP'] = (data['Protocol Type'] == 'ICMP').astype(int)
    data['IPv'] = np.ones(n_samples, dtype=int)  # Assume all IP traffic
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
    
    # Additional features
    data['Magnitue'] = np.random.gamma(2, 100, n_samples)
    data['Radius'] = np.random.gamma(1.5, 50, n_samples)
    data['Covariance'] = np.random.uniform(-1, 1, n_samples)
    data['Variance'] = np.random.gamma(1, 100, n_samples)
    data['Weight'] = np.random.gamma(2, 0.5, n_samples)
    
    # Create labels (0 = Normal, 1-10 = Different attack types)
    # For binary classification: 0 = Normal, 1 = Attack
    attack_prob = 0.15  # 15% attacks
    is_attack = np.random.binomial(1, attack_prob, n_samples)
    
    # Multi-class labels (for reference)
    data['label'] = np.where(is_attack, 
                           np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                         p=[0.3, 0.25, 0.2, 0.15, 0.1]), 
                           0)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure realistic relationships for attacks
    attack_mask = df['label'] > 0
    
    # Attacks tend to have different characteristics
    if attack_mask.sum() > 0:
        df.loc[attack_mask, 'Rate'] *= np.random.uniform(2, 5, attack_mask.sum())
        df.loc[attack_mask, 'fin_flag_number'] *= np.random.uniform(0.1, 3, attack_mask.sum())
        df.loc[attack_mask, 'rst_flag_number'] *= np.random.uniform(2, 8, attack_mask.sum())
        df.loc[attack_mask, 'Tot size'] *= np.random.uniform(0.1, 0.5, attack_mask.sum())
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {n_samples} synthetic IoT samples")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Attack rate: {(df['label'] > 0).mean():.2%}")
    print(f"📏 Shape: {df.shape}")
    print(f"🏷️  Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    # Generate synthetic data
    print("🔄 Starting IoT data generation...")
    df = generate_synthetic_iot_data(n_samples=50000)  # Larger sample for testing
    
    # Create a smaller test set
    test_df = df.sample(n=1000, random_state=42)
    test_df.to_csv('data/raw/test_sample.csv', index=False)
    print("✅ Also created small test sample (1000 rows)")
    print("📂 Files created:")
    print("   - data/raw/synthetic_iot_data.csv (50,000 rows)")
    print("   - data/raw/test_sample.csv (1,000 rows)")
    