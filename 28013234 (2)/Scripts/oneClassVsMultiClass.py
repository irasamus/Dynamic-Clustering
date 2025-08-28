from scapy.all import rdpcap, IP, TCP, UDP
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Deakin mapping
Deakin_mapping = {
    "18:48:be:31:4b:49": 0,
    "70:ee:50:57:95:29": 1,
    "70:09:71:9d:ad:10": 2,
}

def extract_features(packet):
    """Extract features from a packet."""
    features = []
    # Packet length
    features.append(len(packet))
    # Protocol type
    if packet.haslayer(IP):
        features.append(packet[IP].proto)  # IP protocol number
    else:
        features.append(-1)  # No IP layer
    # Source and destination ports
    if packet.haslayer(TCP):
        features.append(packet[TCP].sport)
        features.append(packet[TCP].dport)
    elif packet.haslayer(UDP):
        features.append(packet[UDP].sport)
        features.append(packet[UDP].dport)
    else:
        features.extend([-1, -1])  # No TCP/UDP layers
    # Payload length
    features.append(len(packet.payload))
    return features

features = []
labels = []

packets = rdpcap('../pcapIoT/IoT_2023-07-11.pcap')  

for packet in packets:
    if packet.haslayer("cooked linux"):
        src = packet['cooked linux'].src
        lladdrlen = packet['cooked linux'].lladdrlen
        mac_bytes = src[:lladdrlen]
    
        mac_addr = ':'.join('%02x' % b for b in mac_bytes)
        src_mac = mac_addr.lower()
        if src_mac in Deakin_mapping:
            label = Deakin_mapping[src_mac]
            feature = extract_features(packet)
            features.append(feature)
            labels.append(label)

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=4
)

clf = RandomForestClassifier(random_state=4, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Multiclass Classifier Report:")
print(classification_report(y_test, y_pred))

class_models = {}
for class_label in np.unique(y_train):
    X_class = X_train[y_train == class_label]
    clf_if = IsolationForest(contamination=0.001, random_state=4, n_jobs=-1)
    clf_if.fit(X_class)
    class_models[class_label] = clf_if

predicted_labels = []
for x in X_test:
    scores = {}
    for class_label, clf_if in class_models.items():
        score = clf_if.decision_function([x])[0]
        scores[class_label] = score
    predicted_label = max(scores, key=scores.get)
    predicted_labels.append(predicted_label)

print("Isolation Forest Classifier Report:")
print(classification_report(y_test, predicted_labels))
