import os
import subprocess
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import io
import ipaddress # To help check IP ranges
import sys # To exit gracefully on error
import re # For filename date parsing

# --- Configuration ---
# Using raw strings for Windows paths
PCAP_FILE_TO_ANALYZE = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234\pcapIoT\52095392_IoT_2023-05-16.pcap" # <--- ADJUST IF NEEDED
METADATA_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234 (1)\CSVs"       # <--- ADJUST IF NEEDED
OUTPUT_BASE_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir" # Output directory for the test matrix
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
if not GATEWAY_IP:
    print("Warning: GATEWAY_IP is not set. Categorization will be less accurate.")

# Aggregation metric
AGGREGATION_METRIC = 'count' # Change to 'bytes' and uncomment relevant code if needed

# --- Helper Function: Categorize Destination IP ---
def categorize_destination(ip_str, gateway_ip_str):
    """Categorizes an IP address string."""
    # Categories: Gateway, External, Other Local IP, Broadcast, Multicast, Other/Unknown
    if not isinstance(ip_str, str) or not ip_str:
        return "Non-IP/Invalid"
    try:
        ip_addr = ipaddress.ip_address(ip_str)

        if gateway_ip_str and ip_str == gateway_ip_str: return "Gateway"
        if ip_str == BROADCAST_IP_STR: return "Broadcast"
        # Check common local broadcast
        if ip_str.endswith('.255') and any(ip_addr in net for net in LOCAL_NETWORKS): return "Broadcast"
        if ip_addr.is_multicast: return "Multicast"
        if ip_addr.is_loopback: return "Loopback"
        if ip_addr.is_link_local: return "Link-Local"
        if ip_addr.is_unspecified: return "Unspecified"

        for network in LOCAL_NETWORKS:
            if ip_addr in network:
                return "Other Local IP"

        if ip_addr.is_global:
            return "External"
        else:
            return "Other/Unknown IP"

    except ValueError:
        return "Non-IP/Invalid"

# --- Load Metadata ---
print(f"Loading metadata from {MAC_ADDRESS_FILE}...")
try:
    mac_df = pd.read_csv(MAC_ADDRESS_FILE)
    mac_column_name = 'MAC Address' # Verify this column exists
    if mac_column_name not in mac_df.columns: raise ValueError(f"Column '{mac_column_name}' not found")
    known_macs = mac_df[mac_column_name].str.lower().tolist() # Keep ordered list
    known_macs_set = set(known_macs) # Use set for fast lookups

    mac_to_row_index = {mac: i for i, mac in enumerate(known_macs)}
    num_iot_devices = len(known_macs)

    if num_iot_devices == 0: raise ValueError("No MAC addresses loaded from metadata.")
    print(f"Identified {num_iot_devices} IoT devices.")

except Exception as e:
    print(f"FATAL ERROR loading metadata: {e}")
    sys.exit(1)

# --- Define Output Matrix Structure ---
destination_categories = ["Gateway", "External", "Other Local IP", "Broadcast", "Multicast"]
category_to_col_index = {category: i for i, category in enumerate(destination_categories)}
num_categories = len(destination_categories)

# --- Initialize Output Matrices ---
# Using dictionaries to hold matrices for different layers
matrix_dict_count = {}
# matrix_dict_bytes = {} # Uncomment if calculating bytes

# Initialize Aggregated Matrix
matrix_dict_count['aggregated_ip'] = np.zeros((num_iot_devices, num_categories), dtype=np.int64)
# matrix_dict_bytes['aggregated_ip'] = np.zeros((num_iot_devices, num_categories), dtype=np.int64) # For bytes

# Initialize Layer Matrices (add more or modify as needed)
layer_keys = [
    'external_tcp_tls',
    'external_udp_quic',
    'local_discovery',
    'gateway_dns',
    'other_local_tcp'
]
for key in layer_keys:
    matrix_dict_count[key] = np.zeros((num_iot_devices, num_categories), dtype=np.int64)
    # matrix_dict_bytes[key] = np.zeros((num_iot_devices, num_categories), dtype=np.int64)

# --- Run tshark ---
print(f"\nRunning tshark on {os.path.basename(PCAP_FILE_TO_ANALYZE)}...")
tshark_cmd = [
    'tshark',
    '-r', PCAP_FILE_TO_ANALYZE,
    '-T', 'fields',
    '-e', 'frame.number',
    '-e', 'sll.src.eth',      # Source MAC
    '-e', 'ip.src',
    '-e', 'ip.dst',
    '-e', '_ws.col.protocol', # Protocol Name
    '-e', 'tcp.srcport',      # TCP Ports
    '-e', 'tcp.dstport',
    '-e', 'udp.srcport',      # UDP Ports
    '-e', 'udp.dstport',
    '-e', 'frame.len',        # Frame length for byte counts
    '-E', 'header=y',
    '-E', 'separator=,',
    '-E', 'quote=d',
    '-E', 'occurrence=f',
    '-Y', 'ip and (sll or eth)' # Filter for IP packets with link layer info
]

try:
    process = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
    tshark_output = process.stdout
except Exception as e:
    print(f"FATAL ERROR running tshark: {e}")
    sys.exit(1)

# --- Parse tshark output ---
if not tshark_output or len(tshark_output.splitlines()) <= 1:
    print("FATAL ERROR: No valid IP packet data extracted by tshark.")
    sys.exit(1)

print("Parsing tshark output...")
try:
    df = pd.read_csv(io.StringIO(tshark_output), low_memory=False)
    # Clean source MAC and destination IP
    df['sll.src.eth'] = df['sll.src.eth'].fillna('').astype(str).str.lower()
    df['ip.dst'] = df['ip.dst'].fillna('').astype(str) # Keep as string for categorization
    # Convert ports to numeric, handling errors (NaN if not a number)
    for port_col in ['tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']:
         if port_col in df.columns:
              df[port_col] = pd.to_numeric(df[port_col], errors='coerce') # NaNs if conversion fails
         else:
             print(f"Warning: Column {port_col} not found in tshark output.")
             df[port_col] = np.nan # Add column as NaN if missing

    # Convert frame length
    if 'frame.len' in df.columns:
        df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce').fillna(0).astype(np.int64)
    else:
        print("Warning: Column frame.len not found. Byte aggregation disabled.")
        df['frame.len'] = 0


    print(f"Parsed {len(df)} total IP packets.")
except Exception as e:
    print(f"FATAL ERROR parsing tshark output: {e}")
    sys.exit(1)

# --- Aggregate Data into Matrices ---
print("\nAggregating data into matrices...")
packets_aggregated_total = 0
for _, row in df.iterrows():
    src_mac = row['sll.src.eth']
    dst_ip = row['ip.dst']

    # Check if source is a known IoT device
    row_index = mac_to_row_index.get(src_mac)
    if row_index is not None:
        # Categorize the destination IP
        category = categorize_destination(dst_ip, GATEWAY_IP)
        col_index = category_to_col_index.get(category) # Get index 0-4 or None

        # Only proceed if destination category is one we track
        if col_index is not None:
            protocol = str(row['_ws.col.protocol']).upper()
            dst_port_tcp = row['tcp.dstport'] # Already numeric or NaN
            dst_port_udp = row['udp.dstport'] # Already numeric or NaN
            frame_len = row['frame.len']
            value_to_add_count = 1
            value_to_add_bytes = frame_len # Use pre-processed frame_len

            # --- Aggregate into Overall Matrix ---
            matrix_dict_count['aggregated_ip'][row_index, col_index] += value_to_add_count
            # matrix_dict_bytes['aggregated_ip'][row_index, col_index] += value_to_add_bytes # Uncomment for bytes
            packets_aggregated_total += 1 # Count packets added to *any* matrix

            # --- Aggregate into Layer Matrices ---
            # Layer 1: External TCP/TLS
            if category == "External" and (protocol == "TCP" or "TLS" in protocol):
                matrix_dict_count['external_tcp_tls'][row_index, col_index] += value_to_add_count
                # matrix_dict_bytes['external_tcp_tls'][row_index, col_index] += value_to_add_bytes

            # Layer 2: External UDP/QUIC
            elif category == "External" and (protocol == "UDP" or protocol == "GQUIC" or protocol == "QUIC"):
                matrix_dict_count['external_udp_quic'][row_index, col_index] += value_to_add_count
                # matrix_dict_bytes['external_udp_quic'][row_index, col_index] += value_to_add_bytes

            # Layer 3: Local Discovery (SSDP/MDNS/DHCP -> Bcast/Mcast)
            # Note: DHCP might go to gateway too, adjust if needed
            elif (category == "Broadcast" or category == "Multicast") and \
                 (protocol == "SSDP" or protocol == "MDNS" or protocol == "DHCP"):
                 # Destination column might be Broadcast or Multicast for this layer
                 matrix_dict_count['local_discovery'][row_index, col_index] += value_to_add_count
                 # matrix_dict_bytes['local_discovery'][row_index, col_index] += value_to_add_bytes

            # Layer 4: Gateway Interaction (DNS to Gateway IP)
            elif category == "Gateway" and (protocol == "DNS" or dst_port_udp == 53 or dst_port_tcp == 53):
                 matrix_dict_count['gateway_dns'][row_index, col_index] += value_to_add_count
                 # matrix_dict_bytes['gateway_dns'][row_index, col_index] += value_to_add_bytes

            # Layer 5: Specific Local Interaction (TCP to Other Local IP)
            elif category == "Other Local IP" and protocol == "TCP":
                 matrix_dict_count['other_local_tcp'][row_index, col_index] += value_to_add_count
                 # matrix_dict_bytes['other_local_tcp'][row_index, col_index] += value_to_add_bytes

print(f"Finished aggregation. Processed {packets_aggregated_total} relevant packet entries.")

# --- Save All Result Matrices ---
# Extract date from pcap filename for output filename
file_date = "unknown_date"
match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(PCAP_FILE_TO_ANALYZE))
if match:
    file_date = match.group(1)

# Define header for CSV files (first column name for index)
identifier_column_name = "DeviceIdentifier" # Or "MAC_Address"
csv_column_headers = destination_categories # Just the numerical columns

# Ensure you have the ordered list of MACs for the index
if 'known_macs' not in locals() or len(known_macs) != num_iot_devices:
     print("FATAL ERROR: 'known_macs' list not available or incorrect length during saving.")
     sys.exit(1)

# --- CHANGE: Loop and save using Pandas ---
print("\nSaving matrices using Pandas...")
for layer_key, matrix_data in matrix_dict_count.items():
    # Create specific output directory for this layer/metric
    layer_output_dir = os.path.join(OUTPUT_BASE_DIR, f"layer_{layer_key}_{AGGREGATION_METRIC}")
    os.makedirs(layer_output_dir, exist_ok=True)

    output_filename = f"{file_date}.csv" # Simple date filename within layer folder
    output_path = os.path.join(layer_output_dir, output_filename)

    print(f"  Saving to: {output_path} (Shape: {matrix_data.shape})")
    try:
        # Create DataFrame with MACs as index
        df_to_save = pd.DataFrame(
            matrix_data,
            index=pd.Index(known_macs, name=identifier_column_name), # Use MACs as row index
            columns=csv_column_headers # Use categories as column headers
        )
        # Save using pandas to_csv method
        df_to_save.to_csv(output_path, index=True, header=True)

    except Exception as e:
        print(f"  ERROR saving matrix {output_path} using pandas: {e}")
# --- END CHANGE ---

print("\n--- Script Finished ---")