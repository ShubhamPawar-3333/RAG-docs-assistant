"""
Error Handling Middleware

Provides centralized error handling for the API.
"""

import logging
import time
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler middleware.
    
    Catches all unhandled exceptions and returns proper JSON responses.
    Also logs errors with request context.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log the error with context
            logger.error(
                f"Unhandled error: {str(e)}\n"
                f"Path: {request.url.path}\n"
                f"Method: {request.method}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": str(e),
                    "path": request.url.path,
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware.
    
    Logs all incoming requests with timing information.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}"
        )
        
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    Uses in-memory storage (for production, use Redis).
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests = {}  # IP -> list of timestamps
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health") or request.url.path in ["/ready", "/live"]:
            return await call_next(request)
        
        # Check rate limit
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Clean old requests
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] 
                if t > window_start
            ]
        else:
            self._requests[client_ip] = []
        
        # Check limit
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute}/minute",
                    "retry_after": 60,
                }
            )
        
        # Record request
        self._requests[client_ip].append(current_time)
        
        return await call_next(request)
