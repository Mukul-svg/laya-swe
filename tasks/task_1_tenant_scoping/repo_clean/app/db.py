"""Database Session Manager."""

class DatabaseSession:
    def __init__(self, tenant_id: str = None):
        self.tenant_id = tenant_id
        self._store = {}

    def execute_query(self, query: str, params: dict = None) -> list:
        # In multi-tenant systems, tenant_id is strictly required
        if not self.tenant_id:
            raise ValueError("SECURITY VIOLATION: Database session requires an explicit tenant_id!")
        return [{"tenant_id": self.tenant_id, "query": query, "result": "ok"}]


def get_db(tenant_id: str = None) -> DatabaseSession:
    """Factory for DatabaseSession."""
    return DatabaseSession(tenant_id=tenant_id)
