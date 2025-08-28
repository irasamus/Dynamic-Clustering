import os
import sys
from scapy.all import PcapReader, PcapNgReader

def count_packets_in_pcap(pcap_file):
    count = 0
    try:
        if pcap_file.endswith('.pcapng'):
            reader = PcapNgReader
        else:
            reader = PcapReader

        with reader(pcap_file) as pcap_reader:
            for _ in pcap_reader:
                count += 1
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
    return count

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/pcap/directory")
        sys.exit(1)

    pcap_dir = sys.argv[1]
    if not os.path.isdir(pcap_dir):
        print(f"The directory {pcap_dir} does not exist.")
        sys.exit(1)

    total_packets = 0
    pcap_files = []
    packet_counts = {}

    for root, _, files in os.walk(pcap_dir):
        for filename in files:
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                pcap_file_path = os.path.join(root, filename)
                pcap_files.append(pcap_file_path)

    if not pcap_files:
        print("No pcap or pcapng files found in the directory or its subdirectories.")
        sys.exit(1)

    for pcap_file in pcap_files:
        num_packets = count_packets_in_pcap(pcap_file)
        packet_counts[pcap_file] = num_packets
        total_packets += num_packets
        print(f"{pcap_file}: {num_packets} packets")

    average_packets = total_packets / len(pcap_files)
    print(f"\nTotal number of packets: {total_packets}")
    print(f"Average number of packets per pcap file: {average_packets}")

if __name__ == '__main__':
    main()

