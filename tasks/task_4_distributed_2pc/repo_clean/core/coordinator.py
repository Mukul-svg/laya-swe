"""Distributed Two-Phase Commit (2PC) Coordinator."""
from typing import Dict, List, Any, Optional
from core.ledger import TransactionLedger
from core.participant import ParticipantNode, LockConflictError


class TransactionCoordinator:
    """Coordinates atomic distributed transactions across participant nodes."""

    def __init__(self, participants: List[ParticipantNode], ledger: Optional[TransactionLedger] = None):
        self.participants = participants
        self.ledger = ledger or TransactionLedger()

    def execute_transaction(self, tx_id: str, operations_by_node: Dict[str, Dict[str, Any]]) -> bool:
        """Execute a distributed 2PC transaction.
        
        Args:
            tx_id: Unique transaction identifier.
            operations_by_node: Dict mapping participant node_id to operations dict {key: val}.
            
        Returns:
            True if transaction committed successfully, False if aborted.
        """
        self.ledger.record(tx_id, "PREPARING")

        prepared_nodes: List[ParticipantNode] = []

        # Phase 1: Prepare
        for node in self.participants:
            ops = operations_by_node.get(node.node_id, {})
            if not ops:
                continue

            try:
                vote = node.prepare(tx_id, ops)
                if not vote:
                    # BUG: Participant voted ABORT, but coordinator does NOT abort
                    # the participants that ALREADY prepared and are holding locks!
                    self.ledger.record(tx_id, "ABORTED")
                    return False
                prepared_nodes.append(node)
            except LockConflictError:
                # BUG: Lock conflict occurred, but coordinator does NOT roll back
                # already prepared participants!
                self.ledger.record(tx_id, "ABORTED")
                return False

        # Phase 2: Commit
        self.ledger.record(tx_id, "COMMITTED")
        for node in prepared_nodes:
            node.commit(tx_id)

        return True

    def recover_crashed_transactions(self):
        """Crash recovery procedure.
        
        Inspects the ledger for transactions in flight:
        - If ledger records COMMITTED: ensure all participants commit.
        - If ledger records ABORTED or PREPARING (in-doubt when crashed): ensure all participants abort and release locks.
        """
        # TODO: Implement crash recovery logic
        pass
