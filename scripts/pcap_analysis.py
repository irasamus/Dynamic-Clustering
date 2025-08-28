import os
import subprocess
import pandas as pd
from collections import Counter
import io
import ipaddress # To help check IP ranges
import sys # To exit gracefully on error

# --- Configuration ---
# Using raw strings for Windows paths
PCAP_FILE_TO_ANALYZE = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234\pcapIoT\IoT_2023-08-18.pcap" # <--- ADJUST IF NEEDED
METADATA_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234 (1)\CSVs"       # <--- ADJUST IF NEEDED
MAC_ADDRESS_FILE = os.path.join(METADATA_DIR, "macAddresses.csv")

# Define local network ranges - ADJUST IF YOUR LAB NETWORK WAS DIFFERENT!
LOCAL_NETWORKS = [
    ipaddress.ip_network('192.168.0.0/16', strict=False),
    ipaddress.ip_network('10.0.0.0/8', strict=False),
    ipaddress.ip_network('172.16.0.0/12', strict=False),
]
BROADCAST_IP_STR = '255.255.255.255'

# Set Gateway IP - Manually set based on previous ARP/Traffic analysis
GATEWAY_IP = "192.168.1.1" # <--- VERIFY OR ADJUST

# --- Helper Function: Categorize Destination IP ---
def categorize_destination(ip_str, gateway_ip_str):
    """Categorizes an IP address string."""
    if not isinstance(ip_str, str):
        return "Non-IP/Invalid"
    try:
        ip_addr = ipaddress.ip_address(ip_str)

        if gateway_ip_str and ip_str == gateway_ip_str: return "Gateway"
        if ip_str == BROADCAST_IP_STR: return "Broadcast"
        # Check common local broadcast (adjust if subnet is different than /24)
        if ip_str.endswith('.255') and any(ip_addr in net for net in LOCAL_NETWORKS): return "Broadcast"
        if ip_addr.is_multicast: return "Multicast"
        if ip_addr.is_loopback: return "Loopback"
        if ip_addr.is_link_local: return "Link-Local"
        if ip_addr.is_unspecified: return "Unspecified"

        for network in LOCAL_NETWORKS:
            if ip_addr in network:
                return "Other Local IP" # It's local, but not gateway/bcast/mcast

        # If none of the above and it's a global address, assume External
        if ip_addr.is_global:
            return "External"
        else:
            return "Other/Unknown IP" # Private IPs outside defined local ranges etc.

    except ValueError: # Handle invalid IP strings
        return "Non-IP/Invalid"

# --- Load Metadata ---
print(f"Loading metadata from {MAC_ADDRESS_FILE}...")
mac_to_name = {}
try:
    mac_df = pd.read_csv(MAC_ADDRESS_FILE)
    mac_column_name = 'MAC Address' # Verify this column exists
    if mac_column_name not in mac_df.columns: raise ValueError(f"Column '{mac_column_name}' not found")
    known_macs_set = set(mac_df[mac_column_name].str.lower().tolist())

    device_name_column = 'Device Name' # Verify this column exists
    if device_name_column in mac_df.columns:
        mac_to_name = pd.Series(mac_df[device_name_column].values, index=mac_df[mac_column_name].str.lower()).to_dict()
        print(f"Loaded names from '{device_name_column}' column.")
    else:
        print(f"Warning: Column '{device_name_column}' not found for device names.")

    num_iot_devices = len(known_macs_set)
    if num_iot_devices == 0: raise ValueError("No MAC addresses loaded from metadata.")
    print(f"Identified {num_iot_devices} IoT devices.")

except Exception as e:
    print(f"FATAL ERROR loading metadata: {e}")
    sys.exit(1) # Exit if metadata fails

# --- Run tshark ---
print(f"\nRunning tshark on {os.path.basename(PCAP_FILE_TO_ANALYZE)}...")
tshark_cmd = [
    'tshark',
    '-r', PCAP_FILE_TO_ANALYZE,
    '-T', 'fields',
    '-e', 'frame.number',
    '-e', 'sll.src.eth',
    '-e', 'ip.src',
    '-e', 'ip.dst',
    '-e', '_ws.col.protocol',
    # '-e', 'frame.len', # Can add back if needed for byte counts
    '-E', 'header=y',
    '-E', 'separator=,',
    '-E', 'quote=d',
    '-E', 'occurrence=f',
    '-Y', 'sll or eth' # Capture Layer 2 info
]

try:
    process = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
    tshark_output = process.stdout
except Exception as e:
    print(f"FATAL ERROR running tshark: {e}")
    # Consider printing stderr: print(f"Stderr: {getattr(e, 'stderr', 'N/A')}")
    sys.exit(1) # Exit if tshark fails

# --- Parse tshark output ---
if not tshark_output or len(tshark_output.splitlines()) <= 1:
    print("FATAL ERROR: No packet data extracted by tshark.")
    sys.exit(1)

print("Parsing tshark output...")
try:
    df = pd.read_csv(io.StringIO(tshark_output), low_memory=False)
    # Clean SLL source MAC
    df['sll.src.eth'] = df['sll.src.eth'].fillna('').astype(str).str.lower()
    # Clean IP destination before categorization
    df['ip.dst'] = df['ip.dst'].fillna('').astype(str)
    print(f"Parsed {len(df)} total packets.")
except Exception as e:
    print(f"FATAL ERROR parsing tshark output: {e}")
    sys.exit(1)

# --- Analysis ---
print("\n--- Analysis Results ---")

# 1. Top Protocols Overall
print("\n1. Top Protocols Overall (by packet count):")
protocol_counts = df['_ws.col.protocol'].value_counts()
print(protocol_counts.head(15).to_string()) # Show top 15

# 2. Top IoT Device Source MACs
print("\n2. Top IoT Device Source MAC Address Talkers:")
# Filter for known MACs first, then count
iot_source_macs = df[df['sll.src.eth'].isin(known_macs_set)]['sll.src.eth']
iot_source_mac_counts = Counter(iot_source_macs)
print("MAC Address          Count   Device Name (if known)")
print("-" * 50)
for mac, count in iot_source_mac_counts.most_common(15): # Show top 15 IoT talkers
    device_name = mac_to_name.get(mac, "Unknown")
    print(f"{mac:<20} {count:<7} {device_name}")

# 3. Analysis of IP Packets SENT BY IoT Devices
print("\n3. Analysis of IP Packets SENT BY known IoT Devices:")
# Filter for valid IP packets sent by known devices
iot_sent_ip_traffic = df[
    df['sll.src.eth'].isin(known_macs_set) &
    df['ip.dst'].ne('') # Ensure ip.dst is not empty after fillna/astype(str)
].copy()

# --- Add check for valid parseable IPs before categorization ---
valid_ip_mask = []
for ip_str in iot_sent_ip_traffic['ip.dst']:
    try:
        ipaddress.ip_address(ip_str)
        valid_ip_mask.append(True)
    except ValueError:
        valid_ip_mask.append(False)
iot_sent_ip_traffic = iot_sent_ip_traffic[valid_ip_mask]
# --- End IP validity check ---

total_iot_sent_packets_ip_only = len(iot_sent_ip_traffic)
print(f"Analyzed {total_iot_sent_packets_ip_only} valid IP packets SENT BY known IoT devices:")

if total_iot_sent_packets_ip_only > 0:
    # Apply destination categorization
    iot_sent_ip_traffic['dst_category'] = iot_sent_ip_traffic['ip.dst'].apply(lambda x: categorize_destination(x, GATEWAY_IP))
    destination_counts = iot_sent_ip_traffic['dst_category'].value_counts()
    print("\n  Destination Categories (for these IP packets):")
    print(destination_counts.to_string())

    # Protocols to External destinations
    external_traffic = iot_sent_ip_traffic[iot_sent_ip_traffic['dst_category'] == 'External']
    if not external_traffic.empty:
        iot_ext_protocols = external_traffic['_ws.col.protocol'].value_counts()
        print("\n  Top Protocols used to reach External IPs:")
        print(iot_ext_protocols.head(10).to_string())
    else:
        print("\n  No IP traffic sent to External IPs.")

    # Top External IPs contacted
    if not external_traffic.empty:
        print("\n  Top External IPs contacted:")
        external_ip_counts = Counter(external_traffic['ip.dst']) # Already filtered NaNs
        if external_ip_counts:
            print("IP Address           Count")
            print("-" * 30)
            for ip, count in external_ip_counts.most_common(15):
                print(f"{ip:<20} {count}")
        else: print("  (No valid external IPs counted)")

    # Investigate "Other Local IP" destinations
    other_local_traffic = iot_sent_ip_traffic[iot_sent_ip_traffic['dst_category'] == 'Other Local IP']
    if not other_local_traffic.empty:
         print("\n  Analysis of 'Other Local IP' Destinations (Potential D2D/Local Services):")
         print(f"  Found {len(other_local_traffic)} IP packets sent to specific local IPs (not Gateway/Bcast/Mcast).")

         # Top specific local IPs contacted
         print("\n  Top 'Other Local IP' Destinations contacted:")
         other_local_ip_counts = Counter(other_local_traffic['ip.dst']) # Already filtered NaNs
         if other_local_ip_counts:
            print("IP Address           Count")
            print("-" * 30)
            for ip, count in other_local_ip_counts.most_common(10): # Show top 10
                print(f"{ip:<20} {count}")

            # Protocols used for this traffic
            other_local_protocols = other_local_traffic['_ws.col.protocol'].value_counts()
            print("\n  Protocols used when sending to 'Other Local IP':")
            print(other_local_protocols.head(10).to_string())
         else: print("  (No valid 'Other Local IPs' counted)")
    else:
         print("\n  No IP traffic sent to 'Other Local IP' destinations.")

else:
    print("No valid IP packets sent by known IoT devices found in this file.")

print("\n--- Analysis Complete ---")