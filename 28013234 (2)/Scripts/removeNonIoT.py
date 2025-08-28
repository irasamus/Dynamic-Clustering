import scapy.all as scp
from collections import defaultdict, Counter
import os
import csv
from pathlib import Path

MAC_addresses = {
    "40:f6:bc:bc:89:7b": "Echo Dot (4th Gen)", 	
    "68:3a:48:0d:d4:1c": "Aeotec Smart Hub",
    "70:ee:50:57:95:29": "Netatmo Smart Indoor Security Camera",
    "54:af:97:bb:8d:8f": "TP-Link Tapo Pan/Tilt Wi-Fi Camera",	
    "70:09:71:9d:ad:10": "32' Smart Monitor M80B UHD",
    "00:16:6c:d7:d5:f9": "SAMSUNG Pan/Tilt 1080P Wi-Fi Camera",
    "40:ac:bf:29:04:d4": "EZVIZ Security Camera",
    "10:5a:17:b8:a2:0b": "TOPERSUN Smart Plug",
    "10:5a:17:b8:9f:70": "TOPERSUN Smart Plug",
    "fc:67:1f:53:fa:6e": "Perfk Motion Sensor",
    "1c:90:ff:bf:89:46": "Perfk Motion Sensor",
    "cc:a7:c1:6a:b5:78": "NEST Protect smoke alarm",
    "70:ee:50:96:bb:dc": "Netatmo Weather Station",
    "00:24:e4:e3:15:6e": "Withings Body+ (Scales)",
    "00:24:e4:e4:55:26": "Withings Body+ (Scales)",
    "00:24:e4:f6:91:38": "Withings Connect (Blood Pressure)",
    "00:24:e4:f7:ee:ac": "Withings Connect (Blood Pressure)",
    "70:3a:2d:4a:48:e2": "TUYA Smartdoor Bell",
    "b0:02:47:6f:63:37": "Pix-Star Easy Digital Photo Frame",
    "84:69:93:27:ad:35": "HP Envy",
    "18:48:be:31:4b:49": "Echo Show 8",
    "74:d4:23:32:a2:d7": "Echo Show 8",
    "00:17:88:98:6a:54": "Philips Hue Bridge",
    "6e:fe:2f:5a:d7:7e": "GALAXY Watch5 Pro",
    "90:48:6c:08:da:8a": "Ring Video Doorbell"
}

def already_completed(path):
    if "IoT" in path:
        return True
    else:
        return False
    
def find_pcap_files(directory):
    pcap_files = []
    for root, dirs, files in os.walk(directory):
        if "Output" in dirs:
            dirs.remove("Output")
        if "Data" in dirs:
            dirs.remove("Data")
        for file in files:
            if already_completed(file):
                pass
            elif file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))
    return pcap_files
    

def process_pcap_file(pcap_files, MAC_addresses):
    total_counter = defaultdict(int)
    for pcap_file in pcap_files:
        mac_packets = []
        packet_counts = defaultdict(int)
        for packet in scp.PcapReader(pcap_file):
            if packet.haslayer("cooked linux"):
                eth_src = packet["cooked linux"].src
                hex_mac = ":".join("{:02x}".format(byte) for byte in eth_src)
                mac_address = str(hex_mac)
                mac_address = mac_address[:17]
                if mac_address in MAC_addresses:
                    mac_packets.append(packet)
                    packet_counts[mac_address] += 1
                    total_counter[mac_address] += 1
        if mac_packets:
            mac_output_file = '../IoT_'+os.path.basename(pcap_file)
            scp.wrpcap(mac_output_file, mac_packets)
        print(pcap_file)
        for mac, count in packet_counts.items():
            device_name = MAC_addresses.get(mac, "Unknown Device")
            print(f"{device_name} ({mac}): {count} packets")
    print("Total")
    for mac, count in total_counter.items():
        device_name = MAC_addresses.get(mac, "Unknown Device")
        print(f"{device_name} ({mac}): {count} packets")

directory_path = "../pcapFull"
pcap_files = find_pcap_files(directory_path)
print(pcap_files)
process_pcap_file(pcap_files, MAC_addresses)