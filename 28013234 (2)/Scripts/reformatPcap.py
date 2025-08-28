import os
from scapy.all import PcapReader, PcapWriter
from multiprocessing import Pool

Deakin_mapping = {
"40:f6:bc:bc:89:7b": "Echo Dot (4th Gen)",
"68:3a:48:0d:d4:1c": "Aeotec Smart Hub",
"70:ee:50:57:95:29": "Netatmo Smart Indoor Security Camera",
"54:af:97:bb:8d:8f": "TP-Link Tapo Pan/Tilt Wi-Fi Camera",
"70:09:71:9d:ad:10": "32' Smart Monitor M80B UHD",
"00:16:6c:d7:d5:f9": "SAMSUNG Pan/Tilt 1080P Wi-Fi Camera",
"40:ac:bf:29:04:d4": "EZVIZ Security Camera",
"10:5a:17:b8:a2:0b": "TOPERSUN Smart Plug",
"10:5a:17:b8:9f:70": "TOPERSUN Smart Plug 2",
"fc:67:1f:53:fa:6e": "Perfk Motion Sensor",
"1c:90:ff:bf:89:46": "Perfk Motion Sensor 2",
"cc:a7:c1:6a:b5:78": "NEST Protect smoke alarm",
"70:ee:50:96:bb:dc": "Netatmo Weather Station",
"00:24:e4:e3:15:6e": "Withings Body+ (Scales)",
"00:24:e4:e4:55:26": "Withings Body+ (Scales) 2",
"00:24:e4:f6:91:38": "Withings Connect (Blood Pressure)",
"00:24:e4:f7:ee:ac": "Withings Connect (Blood Pressure) 2",
"70:3a:2d:4a:48:e2": "TUYA Smartdoor Bell",
"b0:02:47:6f:63:37": "Pix-Star Easy Digital Photo Frame",
"84:69:93:27:ad:35": "HP Envy",
"18:48:be:31:4b:49": "Echo Show 8",
"74:d4:23:32:a2:d7": "Echo Show 8 2",
"6e:fe:2f:5a:d7:7e": "GALAXY Watch5 Pro",
"90:48:6c:08:da:8a": "Ring Video Doorbell"
}

def find_pcap_files(directory):
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcap") and "IoT" in file:
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def process_pcap(args):
    pcap_file, mac_addresses, output_base_dir = args
    print(f"Processing {pcap_file}")
    try:
        writers = {}
        for mac in mac_addresses:
            mac_dir = os.path.join(output_base_dir, mac_addresses)
            output_pcap = os.path.join(mac_dir, os.path.basename(pcap_file))
            writers[mac] = PcapWriter(output_pcap, append=True, sync=True)

        with PcapReader(pcap_file) as pcap_reader:
            for packet in pcap_reader:
                if packet.haslayer("cooked linux"):
                    src = packet['cooked linux'].src
                    lladdrlen = packet['cooked linux'].lladdrlen

                    mac_bytes = src[:lladdrlen]
                
                    mac_addr = ':'.join('%02x' % b for b in mac_bytes)
                    src_mac = mac_addr.lower()
                    if src_mac in mac_addresses:
                        writers[src_mac].write(packet)

        for writer in writers.values():
            writer.close()
        print(f"Finished processing {pcap_file}")

    except Exception as e:
        print(f"Error processing {pcap_file}: {e}")


def main():
    pcap_dir = '../pcapIoT'       
    output_base_dir = '../' 

    pcap_files = find_pcap_files(pcap_dir)

    for mac_address in Deakin_mapping.values():
        mac_dir = os.path.join(output_base_dir, mac_address.replace(" ", "_"))
        os.makedirs(mac_dir, exist_ok=True)

    tasks = []
    for pcap_file in pcap_files:
        for mac_address in Deakin_mapping.values():
            tasks.append((pcap_file, mac_address.replace(" ", "_"), output_base_dir))

    with Pool() as pool:
        pool.map(process_pcap, tasks)

if __name__ == '__main__':
    main()
