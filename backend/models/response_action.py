
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ResponseAction(Base):
    __tablename__ = "response_actions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id"), nullable=False, index=True
    )

    recommended_action: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    priority: Mapped[str] = mapped_column(
        String(20), nullable=False
    )

    approval_status: Mapped[str] = mapped_column(
        String(20), default="PENDING", nullable=False
    )

    execution_status: Mapped[str] = mapped_column(
        String(30), default="NOT_EXECUTED", nullable=False
    )

    notes: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
