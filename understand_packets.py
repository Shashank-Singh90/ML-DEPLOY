# This simulates what network data looks like
sample_packet = {
    'timestamp': '2024-01-30 10:30:45',
    'source_ip': '192.168.1.100',      # Your computer
    'dest_ip': '8.8.8.8',               # Google's server
    'source_port': 54321,               # Random high port
    'dest_port': 443,                   # HTTPS port
    'packet_size': 1500,                # Size in bytes
    'tcp_flags': 'SYN',                 # Starting a connection
    'protocol': 'TCP'
}

print("This is what one network packet looks like:")
for key, value in sample_packet.items():
    print(f"  {key}: {value}")

# What makes a packet suspicious?
suspicious_packet = {
    'timestamp': '2024-01-30 10:30:45',
    'source_ip': '192.168.1.100',
    'dest_ip': '192.168.1.1',
    'source_port': 12345,
    'dest_port': 22,                    # SSH port - often attacked
    'packet_size': 60,                  # Very small - maybe a scan
    'tcp_flags': 'SYN',                 # Only SYN, no completion
    'protocol': 'TCP'
}

print("\nThis packet might be suspicious because:")
print("  - It's targeting port 22 (SSH)")
print("  - It's very small (might be scanning)")
print("  - It only has SYN flag (incomplete connection)")
