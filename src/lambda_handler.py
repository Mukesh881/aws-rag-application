"""
Lambda handler for FastAPI application.

This module adapts the FastAPI app to work with AWS Lambda using Mangum.
"""
from mangum import Mangum
from src.app import app

# Mangum adapter: Converts Lambda events to ASGI and back
handler = Mangum(app, lifespan='off')