from data.adapters.cic_ids2017 import CICIDS2017Adapter


sample_row = {
    "Source IP": "192.168.1.25",
    "Destination IP": "192.168.1.10",
    "Source Port": 4500,
    "Destination Port": 80,
    "Protocol": 6,
    "Label": "PortScan"
}


adapter = CICIDS2017Adapter()

alert = adapter.convert(sample_row)

print(alert)