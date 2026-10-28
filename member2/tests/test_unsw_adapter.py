from data.adapters.unsw_nb15 import UNSWNB15Adapter


sample_row = {
    "srcip": "192.168.1.50",
    "dstip": "192.168.1.100",
    "sport": 54321,
    "dsport": 80,
    "proto": "tcp",
    "label": 1,
    "attack_cat": "Reconnaissance"
}


adapter = UNSWNB15Adapter()

alert = adapter.convert(sample_row)

print(alert)