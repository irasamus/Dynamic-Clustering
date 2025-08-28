import pandas as pd
from scapy.all import PcapReader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from collections import Counter
import hashlib
from kan import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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


def is_iot(mac_address):
    mac_address = mac_address.lower()
    return 1 if mac_address in Deakin_mapping else 0

model = KAN(width=[6,2], grid=3, k=3, seed=2024, device=device)

pcap_file = '../pcapFull/2023-08-30.pcap'  

data = []
labels = []


for packet in PcapReader(pcap_file):
    try:
        if packet.haslayer("cooked linux") and packet.haslayer("IP"):
            src = packet['cooked linux'].src
            lladdrlen = packet['cooked linux'].lladdrlen
            mac_bytes = src[:lladdrlen]
        
            mac_addr = ':'.join('%02x' % b for b in mac_bytes)
        else:
            continue

        label = is_iot(mac_addr)
        features = {
            'length': len(packet),
            'protocol': packet.proto if hasattr(packet, 'proto') else 0,
            'ttl': packet.ttl if hasattr(packet, 'ttl') else 0,
            'window_size': packet.window if hasattr(packet, 'window') else 0,
            'dst_port': packet.dport if hasattr(packet, 'dport') else 0,
        }
        if hasattr(packet, 'payload'):
            payload_bytes = bytes(packet.payload)
            payload_hash = hashlib.md5(payload_bytes).hexdigest()
            payload_int = int(payload_hash, 16)
            payload_float = payload_int / float(2**128 - 1)  
            features['payload_float'] = round(payload_float, 12)
        else:
            features['payload_float'] = 0.0
        data.append(features)
        labels.append(label)
    except Exception as e:
        print(f"Error processing packet: {e}")
        continue



dataset = {}
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

dtype = torch.get_default_dtype() 
dataset['train_input'] = torch.from_numpy(X_train_df.to_numpy()).type(dtype).to(device)
dataset['test_input'] = torch.from_numpy(X_test_df.to_numpy()).type(dtype).to(device)
dataset['train_label'] = torch.from_numpy(np.array(y_train)).long().to(device)
dataset['test_label'] = torch.from_numpy(np.array(y_test)).long().to(device)

value_counts = Counter(labels)
print("Frequency of each value:", value_counts)

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

results = model.fit(dataset, opt="LBFGS", steps=30, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())

lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
model.auto_symbolic()
formula1, formula2 = model.symbolic_formula()[0]
print(ex_round(formula1, 4))
print(ex_round(formula2, 4))

def acc(formula1, formula2, X, y):
    batch = X.shape[0]
    correct = 0
    for i in range(batch):
        logit1 = np.array(formula1.subs('x_1', X[i,0]).subs('x_2', X[i,1]).subs('x_3', X[i,2]).subs('x_4', X[i,3]).subs('x_5', X[i,4]).subs('x_6', X[i,5])).astype(np.float64)
        logit2 = np.array(formula2.subs('x_1', X[i,0]).subs('x_2', X[i,1]).subs('x_3', X[i,2]).subs('x_4', X[i,3]).subs('x_5', X[i,4]).subs('x_6', X[i,5])).astype(np.float64)
        if i == 0:
            print(logit1)
            print(logit2)
            print(y[i].item())
        correct += (logit2 > logit1) == y[i].item()
    return correct/batch

print('train acc of the formula:', acc(formula1, formula2, dataset['train_input'], dataset['train_label']))
print('test acc of the formula:', acc(formula1, formula2, dataset['test_input'], dataset['test_label']))