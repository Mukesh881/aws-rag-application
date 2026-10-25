"""
Helpers shared by the API (app.py) and the ingestion pipeline (ingest.py).
"""
from typing import Any, Dict

import boto3
from botocore.config import Config
import structlog


def configure_logging():
    """Configure structlog for JSON output."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def create_bedrock_client(region: str):
    """
    Create a Bedrock runtime client that retries throttling and transient errors.

    Adaptive mode adds client-side rate limiting on top of exponential backoff,
    which matters during bulk ingestion where ThrottlingException is common.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def serialize_index_stats(index_stats) -> Dict[str, Any]:
    """
    Safely serialize Pinecone index stats to avoid JSON serialization issues.

    Args:
        index_stats: Pinecone index stats object

    Returns:
        Dictionary with serializable values
    """
    try:
        stats_dict = {
            "total_vector_count": getattr(index_stats, 'total_vector_count', 0),
            "dimension": getattr(index_stats, 'dimension', 0),
            "index_fullness": getattr(index_stats, 'index_fullness', 0.0),
        }

        # Handle namespaces safely
        namespaces = getattr(index_stats, 'namespaces', {})
        if namespaces:
            # Convert namespace objects to simple dictionaries
            namespace_dict = {}
            for ns_name, ns_obj in namespaces.items():
                if hasattr(ns_obj, 'vector_count'):
                    namespace_dict[ns_name] = {
                        "vector_count": getattr(ns_obj, 'vector_count', 0)
                    }
                else:
                    # Fallback for simple values
                    namespace_dict[ns_name] = str(ns_obj)
            stats_dict["namespaces"] = namespace_dict
        else:
            stats_dict["namespaces"] = {}

        return stats_dict

    except Exception as e:
        structlog.get_logger().warning("Failed to serialize index stats", error=str(e))
        return {"error": "Unable to serialize stats", "raw_type": str(type(index_stats))}
