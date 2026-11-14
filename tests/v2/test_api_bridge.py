from core.v2.api_bridge import _build_response
from core.v2.orchestrator import OrchestratorResponse


def test_build_response_exposes_useragent_coordination_metadata():
    result = OrchestratorResponse(
        message="OK",
        metadata={"coordinated_by": "useragent"},
    )

    response = _build_response(result)

    assert response["response"] == "OK"
    assert response["metadata"]["coordinated_by"] == "useragent"
