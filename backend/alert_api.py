from fastapi import FastAPI
from detection.alert_integration import SecurityAlert

app = FastAPI(
    title="NIDS Alert Receiver",
    description="API for receiving intrusion detection alerts",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "NIDS Alert Receiver is running",
        "status": "active"
    }


@app.post("/api/v1/alerts")
def receive_alert(alert: SecurityAlert):
    print("\n========== SECURITY ALERT RECEIVED ==========")
    print("Alert ID:", alert.alert_id)
    print("Attack Type:", alert.attack_type)
    print("Source IP:", alert.source_ip)
    print("Destination IP:", alert.destination_ip)
    print("Confidence:", alert.confidence)
    print("=============================================\n")

    return {
        "message": "Alert received successfully",
        "alert_id": alert.alert_id,
        "status": "received"
    }