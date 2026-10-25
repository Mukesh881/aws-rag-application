"""
Shared fixtures. All AWS and Pinecone calls are mocked; tests need no network.
"""
import io
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

# Settings are read (and cached) when src.app is imported, so pin the
# environment before any src import and drop variables that change behavior.
for _var in ("PINECONE_API_KEY_SECRET_ID", "API_KEY", "CORS_ALLOW_ORIGINS", "BEDROCK_REGION",
             "BEDROCK_EMBED_MODEL_ID", "EMBED_MODEL_ID", "BEDROCK_LLM_MODEL_ID", "LLM_MODEL_ID"):
    os.environ.pop(_var, None)
os.environ["PINECONE_API_KEY"] = "test-key"
os.environ["AWS_REGION"] = "ap-south-1"

EMBEDDING_DIM = 1024


def bedrock_response(payload: dict) -> dict:
    """Shape a dict like a boto3 invoke_model response."""
    return {"body": io.BytesIO(json.dumps(payload).encode())}


def make_bedrock_client(answer: str = "The answer [Source 1]."):
    """Bedrock mock: Titan-style embeddings and an Anthropic Messages reply."""
    client = mock.Mock()

    def invoke_model(modelId, body, **kwargs):
        if "embed" in modelId:
            return bedrock_response({"embedding": [0.1] * EMBEDDING_DIM})
        return bedrock_response({"content": [{"type": "text", "text": answer}], "stop_reason": "end_turn"})

    client.invoke_model.side_effect = invoke_model
    return client


def make_match(score: float, text: str = "chunk text", source: str = "data/doc.txt", chunk_index: int = 0):
    return SimpleNamespace(score=score, metadata={"text": text, "source": source, "chunk_index": chunk_index})


@pytest.fixture
def mocks():
    """Patch the clients RAGService builds and reset the cached service around each test."""
    from src import app as app_module

    bedrock = make_bedrock_client()
    index = mock.Mock()
    index.query.return_value = SimpleNamespace(matches=[make_match(0.9)])
    index.describe_index_stats.return_value = SimpleNamespace(
        total_vector_count=3, dimension=EMBEDDING_DIM, index_fullness=0.0, namespaces={}
    )
    pinecone = mock.Mock()
    pinecone.return_value.Index.return_value = index

    settings = app_module.get_settings()
    original_api_key = settings.api_key
    app_module.rag_service = None
    with mock.patch.object(app_module, "create_bedrock_client", return_value=bedrock), \
         mock.patch.object(app_module, "Pinecone", pinecone):
        yield SimpleNamespace(bedrock=bedrock, index=index, settings=settings)
    app_module.rag_service = None
    settings.api_key = original_api_key


@pytest.fixture
def client(mocks):
    from fastapi.testclient import TestClient
    from src.app import app

    # No context manager: the startup event doesn't run, as on Lambda
    return TestClient(app)
