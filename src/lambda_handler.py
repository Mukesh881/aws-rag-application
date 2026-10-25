"""
Lambda handler for FastAPI application.

This module adapts the FastAPI app to work with AWS Lambda using Mangum.
"""
from mangum import Mangum
from src.app import app

# Mangum adapter: Converts Lambda events to ASGI and back.
# lifespan='off' means FastAPI startup events never fire here; the RAG
# service is created lazily by src.app.get_rag_service on first request.
handler = Mangum(app, lifespan='off')