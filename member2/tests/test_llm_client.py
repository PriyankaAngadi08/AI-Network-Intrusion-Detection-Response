
from investigation.llm.llm_client import investigate_with_llm

prompt = """
Investigate this sample alert:
Attack type: PortScan
Evidence: A port scan can identify open network ports.
Return JSON with summary, severity, evidence, uncertainties,
recommendations, and sources.
"""

result = investigate_with_llm(prompt)

print(result)

assert isinstance(result, dict)
assert "summary" in result
assert "severity" in result

from investigation.llm.llm_client import investigate_with_llm

prompt = """
Investigate this sample alert:
Attack type: PortScan
Evidence: A port scan can identify open network ports.

Return JSON with summary, severity, evidence,
uncertainties, recommendations, and sources.
"""

result = investigate_with_llm(prompt)

print(result)

assert isinstance(result, dict)
assert "summary" in result
assert "severity" in result

print("LLM investigation test passed!")

print("LLM investigation test passed!")
