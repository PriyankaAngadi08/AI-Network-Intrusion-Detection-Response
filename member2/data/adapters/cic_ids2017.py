from investigation.alert_schema import SecurityAlert
from data.adapters.base_adapter import BaseDatasetAdapter


class CICIDS2017Adapter(BaseDatasetAdapter):

    def convert(self, row) -> SecurityAlert:

        label = str(row.get("Label", "UNKNOWN"))

        if label.upper() == "BENIGN":
            normalized_label = "BENIGN"
            attack_type = "BENIGN"
        else:
            normalized_label = "ATTACK"
            attack_type = label

        return SecurityAlert(
            alert_id="CIC-TEST-001",

            source_ip=row.get("Source IP"),
            destination_ip=row.get("Destination IP"),

            source_port=self.to_int(row.get("Source Port")),
            destination_port=self.to_int(
                row.get("Destination Port")
            ),

            protocol=str(row.get("Protocol"))
            if row.get("Protocol") is not None
            else None,

            attack_type=attack_type,
            label=normalized_label,

            confidence=1.0,

            dataset="CIC-IDS2017"
        )

    @staticmethod
    def to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None