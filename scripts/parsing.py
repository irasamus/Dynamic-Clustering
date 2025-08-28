import os
import subprocess
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import io
import ipaddress # To help check IP ranges
import sys # To exit gracefully on error
import re

# --- Configuration ---
# Using raw strings for Windows paths
PCAP_FILE_TO_ANALYZE = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234\pcapIoT\IoT_2023-07-12.pcap" # <--- ADJUST IF NEEDED
METADATA_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234 (1)\CSVs"       # <--- ADJUST IF NEEDED
OUTPUT_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir" # Output directory for the test matrix
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

# Aggregation metric (for clarity, although only packet count is implemented here)
AGGREGATION_METRIC = 'count'

# --- Helper Function: Categorize Destination IP ---
def categorize_destination(ip_str, gateway_ip_str):
    """Categorizes an IP address string."""
    # Categories: Gateway, External, Other Local IP, Broadcast, Multicast, Other/Unknown
    if not isinstance(ip_str, str) or not ip_str: # Handle NaN/empty after fillna
        return "Non-IP/Invalid"
    try:
        ip_addr = ipaddress.ip_address(ip_str)

        if gateway_ip_str and ip_str == gateway_ip_str: return "Gateway"
        if ip_str == BROADCAST_IP_STR: return "Broadcast"
        # Check common local broadcast (adjust if subnet different than /24)
        if ip_str.endswith('.255') and any(ip_addr in net for net in LOCAL_NETWORKS): return "Broadcast"
        if ip_addr.is_multicast: return "Multicast"
        if ip_addr.is_loopback: return "Loopback" # Should ideally not be a destination
        if ip_addr.is_link_local: return "Link-Local"
        if ip_addr.is_unspecified: return "Unspecified"

        for network in LOCAL_NETWORKS:
            if ip_addr in network:
                return "Other Local IP" # Local, but not gateway/bcast/mcast

        if ip_addr.is_global:
            return "External"
        else:
            return "Other/Unknown IP" # Private IPs outside defined local ranges etc.

    except ValueError: # Handle invalid IP strings like 'ff02::fb' if IPv6 not fully handled elsewhere
        return "Non-IP/Invalid"

# --- Load Metadata ---
print(f"Loading metadata from {MAC_ADDRESS_FILE}...")
try:
    mac_df = pd.read_csv(MAC_ADDRESS_FILE)
    mac_column_name = 'MAC Address' # Verify this column exists
    if mac_column_name not in mac_df.columns: raise ValueError(f"Column '{mac_column_name}' not found")
    known_macs = mac_df[mac_column_name].str.lower().tolist()
    known_macs_set = set(known_macs) # Use set for fast lookups

    # Create mapping from MAC to Row Index (0 to N-1)
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

# Initialize the aggregated matrix for this day
aggregated_matrix = np.zeros((num_iot_devices, num_categories), dtype=np.int64) # Use int for counts

# --- Run tshark ---
print(f"\nRunning tshark on {os.path.basename(PCAP_FILE_TO_ANALYZE)}...")
tshark_cmd = [
    'tshark',
    '-r', PCAP_FILE_TO_ANALYZE,
    '-T', 'fields',
    '-e', 'sll.src.eth', # Source MAC needed for IoT device identification
    '-e', 'ip.dst',      # Destination IP needed for categorization
    # '-e', '_ws.col.protocol', # Not strictly needed for aggregated matrix
    # '-e', 'frame.len',      # Needed only if aggregating bytes
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
    print(f"Parsed {len(df)} total IP packets.")
except Exception as e:
    print(f"FATAL ERROR parsing tshark output: {e}")
    sys.exit(1)

# --- Aggregate Data into Matrix ---
print("\nAggregating data into the matrix...")
packets_aggregated = 0
for _, row in df.iterrows():
    src_mac = row['sll.src.eth']
    dst_ip = row['ip.dst']

    # Check if source is a known IoT device
    row_index = mac_to_row_index.get(src_mac)
    if row_index is not None:
        # Categorize the destination IP
        category = categorize_destination(dst_ip, GATEWAY_IP)

        # Find the column index for the category
        col_index = category_to_col_index.get(category)

        # If the category is one we are tracking, increment the count
        if col_index is not None:
            if AGGREGATION_METRIC == 'count':
                aggregated_matrix[row_index, col_index] += 1
                packets_aggregated += 1
            # Add elif for 'bytes' if implementing later
            # elif AGGREGATION_METRIC == 'bytes':
            #     try:
            #         frame_len = int(row['frame.len']) # Ensure frame.len was extracted
            #         aggregated_matrix[row_index, col_index] += frame_len
            #         packets_aggregated += 1 # Still count packets processed
            #     except (ValueError, KeyError):
            #         print(f"Warning: Could not get frame.len for packet {row.get('frame.number', 'N/A')}")
            #         pass # Skip if frame length isn't available/valid

print(f"Aggregated {packets_aggregated} packets into the matrix.")

# --- Save the Result ---
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract date from pcap filename for output filename
file_date = "unknown_date"
match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(PCAP_FILE_TO_ANALYZE))
if match:
    file_date = match.group(1)

# --- Save the Result ---
# ... (code to create output dir and get filename) ...

# --- CHANGE: Output filename extension ---
output_filename = f"{file_date}_aggregated_{AGGREGATION_METRIC}.csv" # Change extension to .csv
output_path = os.path.join(OUTPUT_DIR, output_filename)

print(f"\nSaving aggregated matrix to: {output_path}")
try:
    # --- CHANGE: Use np.savetxt for CSV ---
    # Define header row based on destination categories
    header_row = ",".join(destination_categories)
    # Save using np.savetxt
    np.savetxt(
        output_path,
        aggregated_matrix,
        delimiter=',',    # Use comma as separator
        fmt='%d',         # Format as integer ('%f' for float if needed)
        header=header_row,# Add the header row
        comments=''       # Prevent adding '#' comment prefix to header
        )
    # --- END CHANGE ---
    print("Save successful.")
    # Optional: Print the matrix shape and sum for verification
    print(f"Matrix shape: {aggregated_matrix.shape}")
    print(f"Total aggregated value (packets): {np.sum(aggregated_matrix)}")

except Exception as e:
    print(f"FATAL ERROR saving matrix: {e}")
    sys.exit(1)

print("\n--- Script Finished ---")