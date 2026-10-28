
from threat_intelligence.mitre_mapping import map_attack

result = map_attack("PortScan")
print(result)

assert result["technique_id"] == "T1046"
assert result["mapping_status"] == "candidate"

unknown = map_attack("UnknownAttack")
print(unknown)

assert unknown["mapping_status"] == "unmapped"

print("MITRE mapping test passed!")
