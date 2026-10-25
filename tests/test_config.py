import json
from unittest import mock

import pytest

from src.config import Settings


def make_settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_bedrock_region_is_independent_of_aws_region(monkeypatch):
    settings = make_settings(monkeypatch, AWS_REGION="ap-south-1")
    assert settings.aws_region == "ap-south-1"
    assert settings.bedrock_region == "us-east-1"

    settings = make_settings(monkeypatch, BEDROCK_REGION="ap-south-1")
    assert settings.bedrock_region == "ap-south-1"


def test_documented_model_id_variables_are_honored(monkeypatch):
    settings = make_settings(
        monkeypatch,
        BEDROCK_LLM_MODEL_ID="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        BEDROCK_EMBED_MODEL_ID="amazon.titan-embed-text-v2:0",
    )
    assert settings.llm_model_id == "apac.anthropic.claude-sonnet-4-20250514-v1:0"
    assert settings.embed_model_id == "amazon.titan-embed-text-v2:0"


def test_legacy_model_id_variable_names_still_work(monkeypatch):
    assert make_settings(monkeypatch, LLM_MODEL_ID="legacy").llm_model_id == "legacy"


@pytest.mark.parametrize("secret_string", [
    "sk-from-secret",
    json.dumps({"PINECONE_API_KEY": "sk-from-secret"}),
    json.dumps({"api_key": "sk-from-secret"}),
])
def test_pinecone_key_loaded_from_secrets_manager(monkeypatch, secret_string):
    monkeypatch.delenv("PINECONE_API_KEY")
    with mock.patch("src.config.boto3.client") as boto_client:
        boto_client.return_value.get_secret_value.return_value = {"SecretString": secret_string}
        settings = make_settings(monkeypatch, PINECONE_API_KEY_SECRET_ID="rag/pinecone")

    assert settings.pinecone_api_key == "sk-from-secret"
    boto_client.assert_called_once_with("secretsmanager", region_name=settings.aws_region)
    boto_client.return_value.get_secret_value.assert_called_once_with(SecretId="rag/pinecone")


def test_direct_pinecone_key_skips_secrets_manager(monkeypatch):
    with mock.patch("src.config.boto3.client") as boto_client:
        settings = make_settings(monkeypatch, PINECONE_API_KEY_SECRET_ID="rag/pinecone")
    assert settings.pinecone_api_key == "test-key"
    boto_client.assert_not_called()


def test_missing_pinecone_key_is_rejected(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY")
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        make_settings(monkeypatch)


def test_cors_origins_parsed_from_comma_separated_list(monkeypatch):
    settings = make_settings(monkeypatch, CORS_ALLOW_ORIGINS="https://a.example, https://b.example")
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
