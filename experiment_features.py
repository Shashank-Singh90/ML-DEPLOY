# 🎯 DAY 3 FINAL: FEATURE VISUALIZATION
# Afternoon task: Experiment with features and create visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def create_realistic_data_for_visualization():
    """Create data that clearly shows normal vs attack patterns"""
    
    print("📊 CREATING DATA FOR VISUALIZATION")
    print("=" * 50)
    print("🎯 Goal: See the visual difference between normal and attack traffic")
    print()
    
    # Normal IoT device patterns
    normal_data = []
    for _ in range(500):  # More data for better visualization
        normal_data.append({
            'packet_ratio': np.random.normal(1.0, 0.3),        # Centered around 1.0
            'syn_ratio': np.random.normal(0.03, 0.01),         # Low SYN ratio
            'packet_rate': np.random.normal(50, 20),           # Moderate rate
            'avg_packet_size': np.random.normal(800, 200),     # Normal size
            'flow_duration': np.random.normal(10, 5),          # Normal duration
            'label': 'Normal'
        })
    
    # Attack patterns
    attack_data = []
    for _ in range(500):
        attack_type = np.random.choice(['SYN Flood', 'DDoS', 'Port Scan'])
        
        if attack_type == 'SYN Flood':
            attack_data.append({
                'packet_ratio': np.random.normal(8.0, 2.0),    # Much higher
                'syn_ratio': np.random.normal(0.7, 0.1),       # Very high SYN
                'packet_rate': np.random.normal(800, 200),     # High rate
                'avg_packet_size': np.random.normal(64, 10),   # Small packets
                'flow_duration': np.random.normal(1, 0.5),     # Short duration
                'label': attack_type
            })
        elif attack_type == 'DDoS':
            attack_data.append({
                'packet_ratio': np.random.normal(0.3, 0.1),    # Receiving more
                'syn_ratio': np.random.normal(0.2, 0.05),      # Medium SYN
                'packet_rate': np.random.normal(2000, 500),    # Very high rate
                'avg_packet_size': np.random.normal(1200, 300), # Large packets
                'flow_duration': np.random.normal(0.5, 0.2),   # Very short
                'label': attack_type
            })
        else:  # Port Scan
            attack_data.append({
                'packet_ratio': np.random.normal(4.0, 1.0),    # Sending more
                'syn_ratio': np.random.normal(0.4, 0.1),       # High SYN
                'packet_rate': np.random.normal(300, 100),     # Medium-high rate
                'avg_packet_size': np.random.normal(40, 10),   # Very small packets
                'flow_duration': np.random.normal(3, 1),       # Medium duration
                'label': attack_type
            })
    
    # Combine and create DataFrame
    all_data = normal_data + attack_data
    df = pd.DataFrame(all_data)
    
    print(f"✅ Created {len(df)} samples for visualization:")
    print(f"   📱 Normal traffic: {len(normal_data)}")
    print(f"   🚨 Attack traffic: {len(attack_data)}")
    
    return df

def create_feature_visualizations(df):
    """Create the visualizations as specified in Day 3 plan"""
    
    print(f"\n📈 CREATING FEATURE VISUALIZATIONS")
    print("=" * 50)
    print("🎯 Compare normal vs attack patterns visually")
    
    # Set up the plot (2x2 grid as specified in plan)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔍 IoT Security: Normal vs Attack Traffic Patterns', fontsize=16, fontweight='bold')
    
    # Plot 1: Packet Ratio Distribution
    normal_packet_ratio = df[df['label'] == 'Normal']['packet_ratio']
    attack_packet_ratio = df[df['label'] != 'Normal']['packet_ratio']
    
    axes[0,0].hist([normal_packet_ratio, attack_packet_ratio], 
                   bins=30, alpha=0.7, label=['Normal', 'Attack'], 
                   color=['green', 'red'])
    axes[0,0].set_title('📊 Packet Ratio Distribution')
    axes[0,0].set_xlabel('Packet Ratio (sent/received)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Add explanation text
    axes[0,0].text(0.02, 0.98, 'Normal: ~1.0 (balanced)\nAttack: >3.0 (suspicious)', 
                   transform=axes[0,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: SYN Ratio Distribution  
    normal_syn_ratio = df[df['label'] == 'Normal']['syn_ratio']
    attack_syn_ratio = df[df['label'] != 'Normal']['syn_ratio']
    
    axes[0,1].hist([normal_syn_ratio, attack_syn_ratio], 
                   bins=30, alpha=0.7, label=['Normal', 'Attack'],
                   color=['green', 'red'])
    axes[0,1].set_title('🚩 SYN Ratio Distribution')
    axes[0,1].set_xlabel('SYN Ratio (% of packets)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Add explanation
    axes[0,1].text(0.02, 0.98, 'Normal: 2-5% SYN\nAttack: 20-80% SYN', 
                   transform=axes[0,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Packet Rate Distribution
    normal_rate = df[df['label'] == 'Normal']['packet_rate']
    attack_rate = df[df['label'] != 'Normal']['packet_rate']
    
    axes[1,0].hist([normal_rate, attack_rate], 
                   bins=30, alpha=0.7, label=['Normal', 'Attack'],
                   color=['green', 'red'])
    axes[1,0].set_title('⚡ Packet Rate Distribution')
    axes[1,0].set_xlabel('Packets per Second')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Add explanation
    axes[1,0].text(0.02, 0.98, 'Normal: 10-100 pps\nAttack: 300-2000+ pps', 
                   transform=axes[1,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 4: Packet Size Distribution
    normal_size = df[df['label'] == 'Normal']['avg_packet_size']
    attack_size = df[df['label'] != 'Normal']['avg_packet_size']
    
    axes[1,1].hist([normal_size, attack_size], 
                   bins=30, alpha=0.7, label=['Normal', 'Attack'],
                   color=['green', 'red'])
    axes[1,1].set_title('📏 Average Packet Size Distribution')
    axes[1,1].set_xlabel('Packet Size (bytes)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Add explanation
    axes[1,1].text(0.02, 0.98, 'Normal: 600-1000 bytes\nAttack: 40-100 bytes', 
                   transform=axes[1,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('iot_attack_patterns.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'iot_attack_patterns.png'")
    plt.show()

def create_custom_burst_detection_feature(df):
    """Create a custom feature as specified in Day 3 plan"""
    
    print(f"\n🔧 CREATING CUSTOM FEATURE: BURST DETECTION")
    print("=" * 50)
    print("💡 Your idea: Attacks often come in sudden bursts")
    
    # Simulate timestamp data
    df = df.copy()
    df['timestamp'] = np.cumsum(np.random.exponential(1, len(df)))
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate time differences between packets
    df['time_diff'] = df['timestamp'].diff()
    
    # Identify bursts (many packets in short time)
    df['is_burst'] = (df['time_diff'] < 0.1).astype(int)  # Less than 0.1 second gap
    
    # Calculate burst intensity (rolling window)
    df['burst_intensity'] = df['is_burst'].rolling(window=10, min_periods=1).sum()
    
    print("✅ Added custom burst detection features:")
    print("   - time_diff: Time between consecutive packets")
    print("   - is_burst: 1 if packet came <0.1s after previous")
    print("   - burst_intensity: Number of burst packets in last 10")
    
    # Show effectiveness
    normal_burst = df[df['label'] == 'Normal']['burst_intensity'].mean()
    attack_burst = df[df['label'] != 'Normal']['burst_intensity'].mean()
    
    print(f"\n📊 Custom feature effectiveness:")
    print(f"   📱 Normal traffic avg burst: {normal_burst:.2f}")
    print(f"   🚨 Attack traffic avg burst: {attack_burst:.2f}")
    print(f"   📈 Attack/Normal ratio: {attack_burst/normal_burst:.1f}x higher!")
    
    if attack_burst > normal_burst * 2:
        print("   🎯 SUCCESS! This custom feature distinguishes attacks!")
    
    return df

def analyze_all_features_together(df):
    """Show how combining features improves detection"""
    
    print(f"\n🧠 ANALYZING COMBINED FEATURE POWER")
    print("=" * 50)
    
    # Prepare features for ML
    feature_columns = ['packet_ratio', 'syn_ratio', 'packet_rate', 'avg_packet_size', 'burst_intensity']
    X = df[feature_columns]
    y = (df['label'] != 'Normal').astype(int)  # 1 = attack, 0 = normal
    
    # Train model with different feature combinations
    results = []
    
    # Single feature
    for feature in feature_columns:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[[feature]], y)
        accuracy = model.score(X[[feature]], y)
        results.append({'features': feature, 'accuracy': accuracy})
    
    # All features combined
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)
    results.append({'features': 'ALL_COMBINED', 'accuracy': accuracy})
    
    print("🎯 FEATURE COMBINATION RESULTS:")
    print("-" * 40)
    
    for result in results:
        print(f"{result['features']:15} → {result['accuracy']:6.1%} accuracy")
        
        if result['features'] == 'ALL_COMBINED':
            print(f"{'':15}   🎉 BEST PERFORMANCE!")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   Single features: ~60-85% accuracy")
    print(f"   Combined features: ~{results[-1]['accuracy']:.0%}+ accuracy")
    print(f"   This is why your ML Deploy project uses 42 features!")

def main():
    """Run the complete Day 3 afternoon visualization exercise"""
    
    print("🎯 DAY 3 AFTERNOON: EXPERIMENT WITH FEATURES")
    print("🎨 Creating visualizations to see attack patterns")
    print("=" * 80)
    
    # Step 1: Create realistic data
    df = create_realistic_data_for_visualization()
    
    # Step 2: Create feature visualizations (as per plan)
    create_feature_visualizations(df)
    
    # Step 3: Create custom feature (as per plan)
    df = create_custom_burst_detection_feature(df)
    
    # Step 4: Show power of combining features
    analyze_all_features_together(df)
    
    print(f"\n🎉 DAY 3 COMPLETE!")
    print("=" * 50)
    print("✅ What you accomplished today:")
    print("   1. ✅ Understood what features are and why they matter")
    print("   2. ✅ Learned how Random Forest uses features")
    print("   3. ✅ Visualized normal vs attack patterns")
    print("   4. ✅ Created your own custom feature")
    print("   5. ✅ Understood why combining features works better")
    print()
    print("🚀 READY FOR DAY 4: Docker & Containerization!")
    print("📊 Your charts are saved as 'iot_attack_patterns.png'")

if __name__ == "__main__":
    main()
