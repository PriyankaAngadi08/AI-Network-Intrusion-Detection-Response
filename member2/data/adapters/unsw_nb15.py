from investigation.alert_schema import SecurityAlert
from data.adapters.base_adapter import BaseDatasetAdapter


class UNSWNB15Adapter(BaseDatasetAdapter):

    def convert(self, row) -> SecurityAlert:

        # UNSW-NB15 uses attack_cat for attack category.
        # The label field normally indicates benign/attack.
        label_value = row.get("label", 0)

        try:
            is_attack = int(label_value) == 1
        except (TypeError, ValueError):
            is_attack = str(label_value).lower() not in {
                "0",
                "benign",
                "normal"
            }

        if is_attack:
            normalized_label = "ATTACK"
            attack_type = str(
                row.get("attack_cat", "UNKNOWN")
            )
        else:
            normalized_label = "BENIGN"
            attack_type = "BENIGN"

        return SecurityAlert(
            alert_id="UNSW-TEST-001",

            source_ip=self.clean_value(row.get("srcip")),
            destination_ip=self.clean_value(row.get("dstip")),

            source_port=self.to_int(row.get("sport")),
            destination_port=self.to_int(row.get("dsport")),

            protocol=self.clean_value(row.get("proto")),

            attack_type=attack_type,
            label=normalized_label,

            confidence=1.0,

            dataset="UNSW-NB15"
        )

    @staticmethod
    def clean_value(value):
        if value is None:
            return None

        return str(value)

    @staticmethod
    def to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None