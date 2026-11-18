
from types import SimpleNamespace
from response.engine import evaluate_incident


test_incident = SimpleNamespace(
    id=3,
    severity="HIGH"
)

result = evaluate_incident(test_incident)

for key, value in result.items():
    print(f"{key}: {value}")
