import os
import re
from datetime import datetime
from scapy.all import PcapReader
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib import rcParams
from collections import OrderedDict
from datetime import timedelta
import matplotlib.dates as mdates
from tqdm import tqdm
import numpy as np
from matplotlib.ticker import MaxNLocator
from brokenaxes import brokenaxes
from matplotlib import gridspec

Deakin_mapping = {
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
    "6e:fe:2f:5a:d7:7e": "GALAXY Watch5 Pro",
    "90:48:6c:08:da:8a": "Ring Video Doorbell"
}

def find_pcap_files(directory):
    pcap_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file.endswith(".pcap"):
            pcap_files.append(file_path)
    return pcap_files

def count_packets_in_pcap(args):
    Deakin_mapping, pcap_file = args
    total_packet_counts = 0
    Non_IoT = []
    Packets_per_file = {}
    unique_macs = set() 
    packet_counts = {mac: 0 for mac in Deakin_mapping.keys()}
    pattern = r'(?:IoT_)?(\d{4}-\d{2}-\d{2})\.pcap'
    filename = os.path.basename(pcap_file)

    match = re.match(pattern, filename)
    date_str = match.group(1) 
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    for pkt in PcapReader(pcap_file):
        total_packet_counts += 1
        if pkt.haslayer("cooked linux"):
            src = pkt['cooked linux'].src
            lladdrlen = pkt['cooked linux'].lladdrlen

            mac_bytes = src[:lladdrlen]
        
            mac_addr = ':'.join('%02x' % b for b in mac_bytes)
            src_mac = mac_addr.lower()

            unique_macs.add(src_mac)

            if src_mac in Deakin_mapping:
                packet_counts[src_mac] += 1
            elif not (src_mac in Non_IoT):
                Non_IoT.append(src_mac)

    Packets_per_file[date_obj] = total_packet_counts
    return (Packets_per_file, packet_counts, Non_IoT, unique_macs)

directory = '../pcapIoT'
pcap_files = find_pcap_files(directory)
print(pcap_files)
total_packet_counts = {mac: 0 for mac in Deakin_mapping.keys()}
mac_file_counts = {mac: [] for mac in Deakin_mapping.keys()}
Non_IoT = []
Packet_counts = []
Packet_per_MAC = {}
Packets_per_file = {}
Unique_macs_per_file = {}  
devices_per_file = {}

args = [(Deakin_mapping, pcap_file) for pcap_file in pcap_files]
with multiprocessing.Pool() as pool:
    results = list(tqdm(
        pool.imap_unordered(count_packets_in_pcap, args),
        total=len(args),
        desc="Processing files",
        unit="file"
    ))
    
for counts, packets, other_devices, unique_macs in results:
    for mac_address, packet_count in packets.items():
        Packet_per_MAC[mac_address] = Packet_per_MAC.get(mac_address, 0) + packet_count
    for date, packet_count in counts.items():
        Packets_per_file[date] = Packets_per_file.get(date, 0) + packet_count
        Unique_macs_per_file[date] = len(unique_macs)  
        devices_per_file[date] = unique_macs
    Non_IoT.extend(other_devices)

Sorted_packets_per_file = OrderedDict(sorted(Packets_per_file.items()))
Sorted_unique_macs_per_file = OrderedDict(sorted(Unique_macs_per_file.items()))
Sorted_devices_per_file = OrderedDict(sorted(devices_per_file.items()))
unique_non_IoT = set(Non_IoT)
print(f"Non-IoT devices: {len(unique_non_IoT)}")

first_date_per_mac = {}
for date in Sorted_devices_per_file.keys():
    macs_on_date = Sorted_devices_per_file[date]
    for mac in macs_on_date:
        if mac not in first_date_per_mac:
            first_date_per_mac[mac] = date

print("First date each MAC address appeared:")
for mac, date in first_date_per_mac.items():
    date_str = date.strftime('%Y-%m-%d')
    print(f"{Deakin_mapping.get(mac, 'Unknown Device')} - {mac} - {date_str}")

print(f"Total packet counts per MAC address: {sum(Packet_per_MAC.values())}")
for key, total_count in Packet_per_MAC.items():
    print(f"Device: {Deakin_mapping[key]}, Total Packets: {total_count}")

average_count = sum(Packets_per_file.values()) / len(Packets_per_file)
print(f"Average number of packets per PCAP file: {average_count:.2f}")
print(f"Total number of packets in the dataset: {sum(Packets_per_file.values())}")

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42
rcParams.update({'font.size': 16})

dates = sorted(Sorted_packets_per_file.keys())
values = [Sorted_packets_per_file[d] for d in dates]

fig = plt.figure(figsize=(20, 8))

bax = brokenaxes(
    xlims=(
        (
            datetime(2023, 5, 13),
            datetime(2023, 5, 30),
        ),
        (
            datetime(2023, 7, 3),
            datetime(2023, 9, 8),
        )
        ,
        (
            datetime(2024, 3, 22),
            datetime(2024, 4, 24),
        )
        ,
        (
            datetime(2024, 5, 3),
            datetime(2024, 5, 30),
        )
    )
)
numeric_dates = mdates.date2num(dates)
bax.bar(dates, values, color='green', width=0.9)
bax.set_yscale('log')
bax.set_xlabel('Time', fontsize = 25, labelpad=50)
bax.set_ylabel('# of Packets', fontsize = 25, labelpad=50)
fig.autofmt_xdate()
plt.tight_layout()
[x.remove() for x in bax.diag_handles]
bax.draw_diags()
fig.savefig("packet_counts_broken_axis.pdf")
plt.close(fig)