"""
Centralized configuration management for the RAG pipeline.

This module uses Pydantic Settings to manage environment variables and
application configuration in a type-safe manner.

Fields are read from the environment variable of the same name
(case-insensitive); fields whose variable name differs declare it with
validation_alias.
"""
import json
from functools import lru_cache
from typing import List, Optional

import boto3
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS Configuration
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)

    # Bedrock Model Configuration
    # Kept separate from AWS_REGION: Lambda sets AWS_REGION to the function's
    # region, which may not serve the configured model / inference profile
    # (e.g. a "us." profile cannot be invoked from ap-south-1).
    bedrock_region: str = Field(
        default="us-east-1",
        validation_alias="BEDROCK_REGION",
        description="Region for Bedrock runtime calls; must match the model ID's inference profile"
    )
    embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        validation_alias=AliasChoices("BEDROCK_EMBED_MODEL_ID", "EMBED_MODEL_ID"),
        description="Bedrock embedding model ID"
    )
    llm_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        validation_alias=AliasChoices("BEDROCK_LLM_MODEL_ID", "LLM_MODEL_ID"),
        description="Bedrock LLM model ID"
    )

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key; leave unset to load it from PINECONE_API_KEY_SECRET_ID"
    )
    pinecone_api_key_secret_id: Optional[str] = Field(
        default=None,
        description="Secrets Manager secret name/ARN holding the Pinecone API key"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Unused by Pinecone client v3+; kept for backwards compatibility"
    )
    pinecone_index_name: str = Field(default="rag-demo-index")

    # Text Processing Configuration
    chunk_size: int = Field(
        default=800,
        description="Number of characters per text chunk"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between consecutive chunks"
    )
    embed_concurrency: int = Field(
        default=8,
        ge=1,
        description="Parallel embedding requests during ingestion"
    )

    # Retrieval Configuration
    top_k: int = Field(
        default=6,
        description="Number of chunks to retrieve for each query"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for retrieved chunks"
    )

    # LLM Generation Configuration
    max_tokens: int = Field(
        default=512,
        description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for LLM generation"
    )

    # Data Source Configuration
    s3_bucket_name: Optional[str] = Field(default=None)
    data_directory: str = Field(
        default="./data",
        description="Local directory for document storage"
    )

    # API Configuration
    api_title: str = Field(default="RAG Pipeline API")
    api_description: str = Field(
        default="A Retrieval-Augmented Generation API using AWS Bedrock and Pinecone"
    )
    api_version: str = Field(default="1.0.0")
    api_key: Optional[str] = Field(
        default=None,
        description="If set, /query and /stats require this value in the X-API-Key header"
    )
    cors_allow_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or *"
    )
    health_cache_seconds: int = Field(
        default=60,
        ge=0,
        description="How long /health reuses its last dependency check"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json or console

    # Development Configuration
    debug: bool = Field(default=False)
    reload: bool = Field(default=False)

    @property
    def cors_origins(self) -> List[str]:
        """Allowed CORS origins as a list."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _resolve_pinecone_api_key(self):
        """Load the Pinecone API key from Secrets Manager when it isn't set directly."""
        if self.pinecone_api_key:
            return self
        if not self.pinecone_api_key_secret_id:
            raise ValueError("Set PINECONE_API_KEY or PINECONE_API_KEY_SECRET_ID")

        secret = boto3.client("secretsmanager", region_name=self.aws_region).get_secret_value(
            SecretId=self.pinecone_api_key_secret_id
        )["SecretString"]
        # Accept either a plain-text secret or a JSON key/value secret
        try:
            parsed = json.loads(secret)
        except json.JSONDecodeError:
            parsed = secret
        if isinstance(parsed, dict):
            parsed = parsed.get("PINECONE_API_KEY") or parsed.get("api_key")
        if not parsed or not isinstance(parsed, str):
            raise ValueError(
                "Secret must be a plain string or JSON with a PINECONE_API_KEY or api_key field"
            )
        self.pinecone_api_key = parsed
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses LRU cache to ensure settings are loaded only once and reused
    throughout the application lifecycle.

    Returns:
        Settings: Configured application settings
    """
    return Settings()


# Convenience function for common use cases
def get_aws_region() -> str:
    """Get the configured AWS region."""
    return get_settings().aws_region


def get_pinecone_config() -> dict:
    """Get Pinecone connection configuration."""
    settings = get_settings()
    return {
        "api_key": settings.pinecone_api_key,
        "index_name": settings.pinecone_index_name
    }


def get_bedrock_config() -> dict:
    """Get Bedrock model configuration."""
    settings = get_settings()
    return {
        "region": settings.bedrock_region,
        "embed_model_id": settings.embed_model_id,
        "llm_model_id": settings.llm_model_id
    }
