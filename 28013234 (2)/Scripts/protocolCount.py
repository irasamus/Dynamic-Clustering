from scapy.all import *
from collections import Counter
import os
import multiprocessing

Deakin_mapping = {
    "40:f6:bc:bc:89:7b": 4,  # Echo Dot (4th Gen)
    "68:3a:48:0d:d4:1c": 7,  # Aeotec Smart Hub
    "70:ee:50:57:95:29": 3,  # Netatmo Smart Indoor Security Camera
    "54:af:97:bb:8d:8f": 3,  # TP-Link Tapo Pan/Tilt Wi-Fi Camera
    "70:09:71:9d:ad:10": 6,  # 32' Smart Monitor M80B UHD
    "00:16:6c:d7:d5:f9": 3,  # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
    "40:ac:bf:29:04:d4": 3,  # EZVIZ Security Camera
    "10:5a:17:b8:a2:0b": 1,  # TOPERSUN Smart Plug
    "10:5a:17:b8:9f:70": 1,  # TOPERSUN Smart Plug
    "fc:67:1f:53:fa:6e": 5,  # Perfk Motion Sensor
    "1c:90:ff:bf:89:46": 5,  # Perfk Motion Sensor
    "cc:a7:c1:6a:b5:78": 5,  # NEST Protect smoke alarm
    "70:ee:50:96:bb:dc": 5,  # Netatmo Weather Station
    "00:24:e4:e3:15:6e": 5,  # Withings Body+ (Scales)
    "00:24:e4:e4:55:26": 5,  # Withings Body+ (Scales)
    "00:24:e4:f6:91:38": 5,  # Withings Connect (Blood Pressure)
    "00:24:e4:f7:ee:ac": 5,  # Withings Connect (Blood Pressure)
    "70:3a:2d:4a:48:e2": 3,  # TUYA Smartdoor Bell
    "b0:02:47:6f:63:37": 6,  # Pix-Star Easy Digital Photo Frame
    "84:69:93:27:ad:35": 6,  # HP Envy
    "18:48:be:31:4b:49": 4,  # Echo Show 8
    "74:d4:23:32:a2:d7": 4,  # Echo Show 8
    "6e:fe:2f:5a:d7:7e": 5,  # GALAXY Watch5 Pro
    "90:48:6c:08:da:8a": 3   # Ring Video Doorbell
}

def get_protocol_pairs(packet):
    """
    Extracts the reversed pairs of protocols from a scapy packet.
    """
    layer = packet.firstlayer()
    protocols = []
    while layer:
        protocols.append(layer.name)
        layer = layer.payload

    protocols = protocols[::-1]
    pairs = []
    for i in range(len(protocols)-1):
        pair = f"{protocols[i]}->{protocols[i+1]}"
        pairs.append(pair)
    return pairs

def find_pcap_files(directory):
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcap") and "IoT" in file:
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def process_pcap_file(pcap_file):
    protocol_pairs_counter = Counter()
    layer_counts = Counter()
    print(f"Processing {pcap_file}")

    for pkt in PcapReader(pcap_file):
        protocol_pairs = get_protocol_pairs(pkt)
        protocol_pairs_counter.update(protocol_pairs)
        current_layer = pkt
        while current_layer:
            layer_type = current_layer.name
            layer_counts[layer_type] += 1
            current_layer = current_layer.payload

    return protocol_pairs_counter, layer_counts



def main():
    pcap_dir = '../pcapIoT'
    files = find_pcap_files(pcap_dir)
    total_protocol_pairs_counter = Counter()
    total_layer_counts = Counter()

    with multiprocessing.Pool() as pool:
        results = pool.map(process_pcap_file, files)

    for protocol_pairs_counter, layer_counts in results:
        total_protocol_pairs_counter.update(protocol_pairs_counter)
        total_layer_counts.update(layer_counts)

    for pair, count in total_protocol_pairs_counter.items():
        print(f"{pair}, Count: {count}")
    
    for layer, count in total_layer_counts.items():
        print(f"{layer}, Count: {count}")

if __name__ == "__main__":
    main()
