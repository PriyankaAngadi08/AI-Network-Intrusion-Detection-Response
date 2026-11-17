
from types import SimpleNamespace

from backend.app.services.investigation_engine import (
    investigate_incident,
)

# Sample incident for testing only
test_incident = SimpleNamespace(
    id=3,
    severity="HIGH",
    attack_type="Port Scan",
)

report = investigate_incident(test_incident)

print("=== INVESTIGATION REPORT TEST ===")
print("Summary:", report["summary"])
print("Analysis type:", report["analysis_type"])
print("Evidence:", report["evidence"])
print("Uncertainties:", report["uncertainties"])
print("Recommendations:", report["recommendations"])

# Basic checks
assert report["analysis_type"] == "RULE_BASED_PRELIMINARY"
assert report["evidence"][0]["detail"]["incident_id"] == 3
assert report["uncertainties"]
assert report["recommendations"]

print("\nTEST PASSED")
