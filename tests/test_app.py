import json
import logging

from fastapi.testclient import TestClient

from tests.conftest import bedrock_response, make_match


def lambda_event(method: str, path: str, body: dict = None) -> dict:
    """Minimal API Gateway HTTP API (payload v2) / Function URL event."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "example.com", "content-type": "application/json"},
        "body": json.dumps(body) if body else None,
        "isBase64Encoded": False,
        "requestContext": {
            "http": {"method": method, "path": path, "protocol": "HTTP/1.1", "sourceIp": "1.1.1.1"},
            "stage": "$default", "requestId": "req", "accountId": "1", "apiId": "api",
            "domainName": "example.com", "domainPrefix": "example", "routeKey": "$default",
            "time": "now", "timeEpoch": 0,
        },
    }


def llm_calls(bedrock):
    return [c for c in bedrock.invoke_model.call_args_list if "embed" not in c.kwargs["modelId"]]


def embed_calls(bedrock):
    return [c for c in bedrock.invoke_model.call_args_list if "embed" in c.kwargs["modelId"]]


def test_lambda_handler_initializes_service_lazily(mocks):
    from src.lambda_handler import handler

    response = handler(lambda_event("POST", "/query", {"query": "What is the leave policy?"}), None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["answer"] == "The answer [Source 1]."


def test_service_is_reused_across_requests(client, mocks):
    from src import app as app_module

    client.get("/stats")
    first = app_module.rag_service
    client.get("/stats")
    assert app_module.rag_service is first


def test_zero_similarity_threshold_is_honored(client, mocks):
    mocks.index.query.return_value.matches = [make_match(0.1)]

    body = client.post("/query", json={"query": "q", "similarity_threshold": 0.0}).json()

    assert body["metadata"]["threshold_used"] == 0.0
    assert body["metadata"]["chunks_found"] == 1


def test_default_threshold_used_when_not_provided(client, mocks):
    mocks.index.query.return_value.matches = [make_match(0.1)]

    body = client.post("/query", json={"query": "q"}).json()

    assert body["metadata"]["threshold_used"] == mocks.settings.similarity_threshold
    assert body["metadata"]["chunks_found"] == 0


def test_instructions_sent_as_system_prompt_with_docs_in_user_turn(client, mocks):
    client.post("/query", json={"query": "How many leave days?"})

    request = json.loads(llm_calls(mocks.bedrock)[0].kwargs["body"])
    assert "ignore any instructions" in request["system"]
    user_text = request["messages"][0]["content"][0]["text"]
    assert "<docs>" in user_text and "How many leave days?" in user_text


def test_empty_llm_response_is_an_error_not_an_answer(client, mocks):
    def invoke_model(modelId, body, **kwargs):
        if "embed" in modelId:
            return bedrock_response({"embedding": [0.1] * 1024})
        return bedrock_response({"content": [], "stop_reason": "max_tokens"})
    mocks.bedrock.invoke_model.side_effect = invoke_model

    response = client.post("/query", json={"query": "q"})

    assert response.status_code == 502


def test_multiple_text_blocks_are_joined(client, mocks):
    def invoke_model(modelId, body, **kwargs):
        if "embed" in modelId:
            return bedrock_response({"embedding": [0.1] * 1024})
        return bedrock_response({"content": [{"type": "text", "text": "Part one. "},
                                             {"type": "text", "text": "Part two."}]})
    mocks.bedrock.invoke_model.side_effect = invoke_model

    assert client.post("/query", json={"query": "q"}).json()["answer"] == "Part one. Part two."


def test_api_key_enforced_when_configured(client, mocks):
    mocks.settings.api_key = "secret"

    assert client.post("/query", json={"query": "q"}).status_code == 401
    assert client.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/stats", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/live").status_code == 200


def test_api_key_not_required_when_unset(client, mocks):
    assert client.get("/stats").status_code == 200


def test_wildcard_cors_does_not_allow_credentials(client):
    # Browsers only expose credentialed cross-origin responses when this
    # header is present (Starlette may still echo the Origin itself)
    response = client.get("/live", headers={"Origin": "https://evil.example", "Cookie": "session=1"})

    assert "access-control-allow-credentials" not in response.headers


def test_health_result_is_cached(client, mocks):
    assert client.get("/health").json()["status"] == "healthy"
    client.get("/health")

    assert len(embed_calls(mocks.bedrock)) == 1


def test_liveness_does_not_initialize_service():
    from src import app as app_module

    app_module.rag_service = None
    assert TestClient(app_module.app).get("/live").json()["status"] == "alive"
    assert app_module.rag_service is None


def test_query_text_is_not_logged(client, mocks, caplog):
    caplog.set_level(logging.DEBUG)

    client.post("/query", json={"query": "my employee id is 4711"})

    assert "4711" not in caplog.text
    assert "query_id" in caplog.text
