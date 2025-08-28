import os
import subprocess
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import io
import ipaddress # To help check IP ranges
import sys # To exit gracefully on error
import re # For filename date parsing
import glob # To find all pcap files
import time # For timing

# --- Configuration ---
# Using raw strings for Windows paths
PCAP_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234\pcapIoT"         # <--- ADJUST IF NEEDED
METADATA_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234 (1)\CSVs"       # <--- ADJUST IF NEEDED
OUTPUT_BASE_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir" # Output directory for the test matrix
MAC_ADDRESS_FILE = os.path.join(METADATA_DIR, "macAddresses.csv")

# --- Define the start date for processing ---
START_PROCESSING_DATE = "2023-05-27" # Files before this date will be skipped

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
AGGREGATION_METRIC = 'count' # Change to 'bytes' and adjust aggregation logic if needed

# --- Helper Function: Categorize Destination IP ---
def categorize_destination(ip_str, gateway_ip_str):
    """Categorizes an IP address string."""
    if not isinstance(ip_str, str) or not ip_str:
        return "Non-IP/Invalid"
    try:
        ip_addr = ipaddress.ip_address(ip_str)
        if gateway_ip_str and ip_str == gateway_ip_str: return "Gateway"
        if ip_str == BROADCAST_IP_STR: return "Broadcast"
        if ip_str.endswith('.255') and any(ip_addr in net for net in LOCAL_NETWORKS): return "Broadcast"
        if ip_addr.is_multicast: return "Multicast"
        if ip_addr.is_loopback: return "Loopback"
        if ip_addr.is_link_local: return "Link-Local"
        if ip_addr.is_unspecified: return "Unspecified"
        for network in LOCAL_NETWORKS:
            if ip_addr in network: return "Other Local IP"
        if ip_addr.is_global: return "External"
        else: return "Other/Unknown IP"
    except ValueError:
        return "Non-IP/Invalid"

# --- Load Metadata (Once Before Loop) ---
print(f"Loading metadata from {MAC_ADDRESS_FILE}...")
mac_to_name = {}
try:
    mac_df = pd.read_csv(MAC_ADDRESS_FILE)
    mac_column_name = 'MAC Address' # Verify this column exists
    if mac_column_name not in mac_df.columns: raise ValueError(f"Column '{mac_column_name}' not found")
    known_macs = mac_df[mac_column_name].str.lower().tolist() # Ordered list needed for saving
    known_macs_set = set(known_macs)
    mac_to_row_index = {mac: i for i, mac in enumerate(known_macs)}
    num_iot_devices = len(known_macs)

    device_name_column = 'Device Name' # Verify this column exists
    if device_name_column in mac_df.columns:
        mac_to_name = pd.Series(mac_df[device_name_column].values, index=mac_df[mac_column_name].str.lower()).to_dict()
        print(f"Loaded names from '{device_name_column}' column.")
    else:
        print(f"Warning: Column '{device_name_column}' not found for device names.")

    if num_iot_devices == 0: raise ValueError("No MAC addresses loaded from metadata.")
    print(f"Identified {num_iot_devices} IoT devices.")

except Exception as e:
    print(f"FATAL ERROR loading metadata: {e}")
    sys.exit(1)

# --- Define Output Matrix Structure (Once Before Loop) ---
destination_categories = ["Gateway", "External", "Other Local IP", "Broadcast", "Multicast"]
category_to_col_index = {category: i for i, category in enumerate(destination_categories)}
num_categories = len(destination_categories)

layer_keys = [ # Layers for which matrices will be generated
    'aggregated_ip',
    'external_tcp_tls',
    'external_udp_quic',
    'local_discovery',
    'gateway_dns',
    'other_local_tcp'
]

# --- Find and Sort PCAP Files ---
print(f"\nScanning for pcap files in {PCAP_DIR}...")
pcap_files_info = []
try:
    potential_files = glob.glob(os.path.join(PCAP_DIR, "*.pcap")) + \
                      glob.glob(os.path.join(PCAP_DIR, "*.pcapng"))
    for filepath in potential_files:
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            pcap_files_info.append({"path": filepath, "date": match.group(1), "filename": filename})
        else: print(f"Warning: Could not parse date from filename: {filename}. Skipping.")
    if not pcap_files_info: raise FileNotFoundError(f"No pcap files with parsable dates found in {PCAP_DIR}.")
    pcap_files_info.sort(key=lambda x: x['date'])
    print(f"Found {len(pcap_files_info)} pcap files to process.")
except Exception as e:
    print(f"FATAL ERROR finding pcap files: {e}")
    sys.exit(1)

# --- Main Processing Loop ---
total_files = len(pcap_files_info)
start_time_total = time.time()
processed_count = 0 # Keep track of files actually processed
skipped_count = 0
print(f"\n--- Starting processing loop for {total_files} files, beginning from date {START_PROCESSING_DATE} ---")

for i, file_info in enumerate(pcap_files_info):
    pcap_file_to_analyze = file_info['path']
    file_date = file_info['date']
    filename = file_info['filename']

    # --- Skip files before the start date ---
    if file_date < START_PROCESSING_DATE:
        skipped_count += 1
        continue # Go to the next file in the loop
    # --- End Skip Check ---

    processed_count += 1 # Increment count only if not skipped
    print(f"\nProcessing file {i+1}/{total_files} (Actual Processed: {processed_count}): {filename} (Date: {file_date})")
    start_time_file = time.time()

    # Initialize Output Matrices for THIS DAY
    matrix_dict_count = {key: np.zeros((num_iot_devices, num_categories), dtype=np.int64) for key in layer_keys}

    # Run tshark
    tshark_cmd = [
        'tshark','-r', pcap_file_to_analyze, '-T', 'fields',
        '-e', 'sll.src.eth','-e', 'ip.dst','-e', '_ws.col.protocol',
        '-e', 'tcp.dstport','-e', 'udp.dstport','-e', 'frame.len',
        '-E', 'header=y', '-E', 'separator=,', '-E', 'quote=d', '-E', 'occurrence=f',
        '-Y', 'ip and (sll or eth)'
    ]
    try:
        process = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        tshark_output = process.stdout
        if not tshark_output or len(tshark_output.splitlines()) <= 1:
             print(f"  Warning: No valid IP packet data extracted by tshark for {filename}. Skipping aggregation.")
             continue
    except Exception as e:
        print(f"  ERROR running tshark on {filename}: {e}. Skipping file.")
        continue

    # Parse tshark output
    try:
        df = pd.read_csv(io.StringIO(tshark_output), low_memory=False)
        df['sll.src.eth'] = df['sll.src.eth'].fillna('').astype(str).str.lower()
        df['ip.dst'] = df['ip.dst'].fillna('').astype(str)
        for port_col in ['tcp.dstport', 'udp.dstport']:
             if port_col in df.columns: df[port_col] = pd.to_numeric(df[port_col], errors='coerce')
             else: df[port_col] = np.nan
        if 'frame.len' in df.columns: df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce').fillna(0).astype(np.int64)
        else: df['frame.len'] = 0
    except Exception as e:
        print(f"  ERROR parsing tshark output for {filename}: {e}. Skipping file.")
        continue

    # Aggregate Data into Matrices
    packets_aggregated_this_file = 0
    for _, row in df.iterrows():
        src_mac = row['sll.src.eth']
        dst_ip = row['ip.dst']
        row_index = mac_to_row_index.get(src_mac)

        if row_index is not None: # If source is a known IoT device
            category = categorize_destination(dst_ip, GATEWAY_IP)
            col_index = category_to_col_index.get(category)

            if col_index is not None: # If destination category is one we track
                protocol = str(row['_ws.col.protocol']).upper()
                dst_port_tcp = row['tcp.dstport']
                dst_port_udp = row['udp.dstport']
                value_to_add_count = 1

                # Aggregate into Overall Matrix
                matrix_dict_count['aggregated_ip'][row_index, col_index] += value_to_add_count
                packets_aggregated_this_file += 1

                # Aggregate into Layer Matrices
                is_tcp_tls = "TCP" in protocol or "TLS" in protocol
                is_udp_quic = protocol == "UDP" or "QUIC" in protocol
                is_dns = protocol == "DNS" or (dst_port_udp == 53.0) or (dst_port_tcp == 53.0)
                is_discovery = protocol == "SSDP" or protocol == "MDNS" or protocol == "DHCP"

                if category == "External" and is_tcp_tls:
                    matrix_dict_count['external_tcp_tls'][row_index, col_index] += value_to_add_count
                elif category == "External" and is_udp_quic:
                    matrix_dict_count['external_udp_quic'][row_index, col_index] += value_to_add_count
                elif (category == "Broadcast" or category == "Multicast") and is_discovery:
                     matrix_dict_count['local_discovery'][row_index, col_index] += value_to_add_count
                elif category == "Gateway" and is_dns:
                     matrix_dict_count['gateway_dns'][row_index, col_index] += value_to_add_count
                elif category == "Other Local IP" and protocol == "TCP":
                     matrix_dict_count['other_local_tcp'][row_index, col_index] += value_to_add_count

    # --- Save All Result Matrices for THIS DAY using Pandas ---
    identifier_column_name = "MAC_Address"
    csv_column_headers = destination_categories

    print(f"  Saving matrices for date {file_date}...")
    for layer_key, matrix_data in matrix_dict_count.items():
        layer_output_dir = os.path.join(OUTPUT_BASE_DIR, f"layer_{layer_key}_{AGGREGATION_METRIC}")
        # --- Removed os.makedirs --- assumes folder exists ---
        output_filename = f"{file_date}.csv"
        output_path = os.path.join(layer_output_dir, output_filename)

        # Optional safety check before saving
        if not os.path.isdir(layer_output_dir):
             print(f"  ERROR: Output directory does not exist: {layer_output_dir}. Skipping save for layer {layer_key}.")
             continue # Skip saving this layer if folder missing

        try:
            df_to_save = pd.DataFrame(
                matrix_data,
                index=pd.Index(known_macs, name=identifier_column_name),
                columns=csv_column_headers
            )
            df_to_save.to_csv(output_path, index=True, header=True)
        except Exception as e:
            print(f"  ERROR saving matrix {output_path} using pandas: {e}")

    # --- End of Day Processing ---
    file_duration = time.time() - start_time_file
    print(f"  Finished processing {filename} in {file_duration:.2f}s. Aggregated {packets_aggregated_this_file} packet entries.")

# --- End Main Processing Loop ---
total_duration = time.time() - start_time_total
print(f"\n--- Completed processing. Skipped {skipped_count} files before {START_PROCESSING_DATE}. Processed {processed_count} files in {total_duration:.2f}s ---")