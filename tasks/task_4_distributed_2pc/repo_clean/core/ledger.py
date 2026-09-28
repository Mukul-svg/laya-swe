"""Persistent Transaction Ledger for 2PC."""
from typing import Dict, List, Optional


class TransactionLedger:
    """Records distributed transaction states across coordinator lifecycles."""

    def __init__(self):
        self._entries: Dict[str, str] = {}
        self._history: List[Dict[str, str]] = []

    def record(self, tx_id: str, status: str):
        """Record status transition for a transaction: PREPARING, COMMITTED, ABORTED."""
        valid_statuses = {"PREPARING", "COMMITTED", "ABORTED"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid transaction status: {status}")
        self._entries[tx_id] = status
        self._history.append({"tx_id": tx_id, "status": status})

    def get_status(self, tx_id: str) -> Optional[str]:
        """Return the current recorded status of tx_id."""
        return self._entries.get(tx_id)

    def get_all_transactions(self) -> Dict[str, str]:
        """Return snapshot of all transactions and their recorded terminal status."""
        return dict(self._entries)

    def clear(self):
        """Clear the ledger."""
        self._entries.clear()
        self._history.clear()
