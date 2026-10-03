import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock database simulating enterprise document clearance requirements
DOCUMENT_CLEARANCES = {
    "NDA-402": "clearance_level_1",
    "SEC_10K_2025": "clearance_level_2",
    "HR_POLICY_GENERAL": "clearance_level_1",
    "IT_RUNBOOK_CORE_INFRA": "clearance_level_3"
}

# Mock RBAC Mapping (In production, this interfaces with Active Directory/OAuth)
ROLE_CLEARANCES = {
    "standard_employee": ["clearance_level_1"],
    "finance_manager": ["clearance_level_1", "clearance_level_2"],
    "infrastructure_admin": ["clearance_level_1", "clearance_level_2", "clearance_level_3"]
}

def decode_identity_token(jwt_token: str) -> str:
    """
    Cryptographically verifies the user token.
    (Mocked for simulation purposes)
    """
    # E.g., jwt.decode(jwt_token, PUBLIC_KEY, algorithms=["RS256"])
    mock_token_payloads = {
        "token_emp_123": "standard_employee",
        "token_mgr_456": "finance_manager",
        "token_admin_789": "infrastructure_admin"
    }
    return mock_token_payloads.get(jwt_token, "unauthorized")

def verify_document_access(user_role: str, document_id: str) -> bool:
    """
    Verifies if the assigned user role has clearance for the requested document.
    """
    allowed_levels = ROLE_CLEARANCES.get(user_role, [])
    required_level = DOCUMENT_CLEARANCES.get(document_id, "clearance_level_3") # Default to highest security
    
    return required_level in allowed_levels

def filter_vector_results(user_token: str, retrieved_chunks: list) -> list:
    """
    Intercepts the HNSW graph search results and filters out any document chunks 
    the user does not have explicit clearance to view, entirely preventing them 
    from entering the LLM's context window.
    """
    user_role = decode_identity_token(user_token)
    if user_role == "unauthorized":
        logger.warning("Unauthorized access attempt intercepted.")
        return []

    secure_context = []
    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id")
        if verify_document_access(user_role, doc_id):
            secure_context.append(chunk)
        else:
            logger.warning(f"RBAC Blocked: User '{user_role}' denied access to '{doc_id}'")
            
    return secure_context

# --- Simulation Example ---
if __name__ == "__main__":
    # Simulated retrieved chunks from HNSW Database
    raw_retrieval = [
        {"doc_id": "NDA-402", "text": "Penalty for breach is $50,000."},
        {"doc_id": "SEC_10K_2025", "text": "Q3 Revenue projected at $45M."},
        {"doc_id": "IT_RUNBOOK_CORE_INFRA", "text": "Admin password is: root123"}
    ]
    
    # Simulate a standard employee querying the system
    print("Testing Standard Employee Token:")
    safe_chunks = filter_vector_results("token_emp_123", raw_retrieval)
    print(f"Allowed Context: {[c['doc_id'] for c in safe_chunks]}\n")
