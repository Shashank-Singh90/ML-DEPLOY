# 🎯 SIMPLE FEATURE UNDERSTANDING DEMO
# Let's understand features step by step!

print("🎓 LEARNING: What are ML features?")
print("=" * 50)

# SCENARIO: You're monitoring your home IoT devices
print("\n📱 SCENARIO: Monitoring your smart home devices")
print("-" * 30)

# Raw data from your smart doorbell
smart_doorbell = {
    "device_name": "Smart Doorbell",
    "packets_sent": 20,      # Doorbell sent 20 packets
    "packets_received": 25,  # Doorbell received 25 packets  
    "time_period": 10,       # Over 10 seconds
    "syn_flags": 1,          # 1 SYN flag (normal connection start)
    "data_size": 5000        # 5000 bytes total
}

print("📊 Raw data from smart doorbell:")
for key, value in smart_doorbell.items():
    print(f"   {key}: {value}")

print("\n🔧 Now let's create FEATURES (the magic numbers):")
print("-" * 30)

# Feature 1: Packet ratio (how much device sends vs receives)
packet_ratio = smart_doorbell["packets_sent"] / smart_doorbell["packets_received"]
print(f"📊 packet_ratio = {smart_doorbell['packets_sent']} ÷ {smart_doorbell['packets_received']} = {packet_ratio:.2f}")

if packet_ratio < 1:
    print("   💭 Device receives more than it sends (normal for cameras)")
elif packet_ratio > 3:
    print("   🚨 Device sends way more than it receives (SUSPICIOUS!)")
else:
    print("   ✅ Balanced traffic (normal)")

# Feature 2: Packet rate (how fast is the communication)
total_packets = smart_doorbell["packets_sent"] + smart_doorbell["packets_received"]
packet_rate = total_packets / smart_doorbell["time_period"]
print(f"\n📊 packet_rate = {total_packets} ÷ {smart_doorbell['time_period']} = {packet_rate:.1f} packets/second")

if packet_rate > 100:
    print("   🚨 Very high rate (possible DDoS attack)")
elif packet_rate > 50:
    print("   ⚠️  High rate (monitor this device)")
else:
    print("   ✅ Normal rate for smart home device")

# Feature 3: SYN ratio (connection attempt pattern)
syn_ratio = smart_doorbell["syn_flags"] / total_packets
print(f"\n📊 syn_ratio = {smart_doorbell['syn_flags']} ÷ {total_packets} = {syn_ratio:.3f} ({syn_ratio*100:.1f}%)")

if syn_ratio > 0.5:
    print("   🚨 ATTACK! Too many connection attempts (SYN flood)")
elif syn_ratio > 0.1:
    print("   ⚠️  Elevated SYN ratio (keep watching)")
else:
    print("   ✅ Normal connection pattern")

print("\n🎯 SUMMARY: Features turn raw data into meaningful numbers")
print("   Raw: 'Device sent packets'")
print("   Feature: 'packet_ratio = 0.80 (normal pattern)'")

print("\n💡 Your ML model uses these feature numbers to make decisions!")
print("   High syn_ratio = Attack")
print("   Normal packet_ratio = Safe")
print("   Combine ALL features = Final decision")