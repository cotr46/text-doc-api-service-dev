"""
Authentication and Authorization Module for Text Analysis API
Provides API key authentication, audit logging, and rate limiting
"""

import os
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

# Configure audit logger
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# Create handler if not exists
if not audit_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

class AuthConfig:
    """Authentication configuration"""
    
    # API Keys (in production, these would be stored securely)
    VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()
    
    # Rate limiting configuration
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per window
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # window in seconds (1 hour)
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))  # burst limit per minute
    
    # Authentication settings
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    AUDIT_SENSITIVE_OPERATIONS = True

class RateLimiter:
    """
    Rate limiting implementation with sliding window and burst protection
    """
    
    def __init__(self):
        self.requests = defaultdict(deque)  # client_id -> deque of timestamps
        self.burst_requests = defaultdict(deque)  # client_id -> deque of recent requests
    
    def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limits
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        now = time.time()
        
        # Clean old requests (sliding window)
        self._cleanup_old_requests(client_id, now)
        
        # Check burst limit (per minute)
        burst_window = now - 60  # 1 minute
        burst_requests = self.burst_requests[client_id]
        
        # Remove old burst requests
        while burst_requests and burst_requests[0] < burst_window:
            burst_requests.popleft()
        
        if len(burst_requests) >= AuthConfig.RATE_LIMIT_BURST:
            return False, {
                "error": "Burst rate limit exceeded",
                "limit": AuthConfig.RATE_LIMIT_BURST,
                "window": "1 minute",
                "retry_after": 60 - (now - burst_requests[0])
            }
        
        # Check main rate limit (per hour)
        requests = self.requests[client_id]
        
        if len(requests) >= AuthConfig.RATE_LIMIT_REQUESTS:
            oldest_request = requests[0]
            retry_after = AuthConfig.RATE_LIMIT_WINDOW - (now - oldest_request)
            
            return False, {
                "error": "Rate limit exceeded",
                "limit": AuthConfig.RATE_LIMIT_REQUESTS,
                "window": f"{AuthConfig.RATE_LIMIT_WINDOW} seconds",
                "retry_after": retry_after
            }
        
        # Record the request
        requests.append(now)
        burst_requests.append(now)
        
        return True, {
            "remaining": AuthConfig.RATE_LIMIT_REQUESTS - len(requests),
            "reset_time": now + AuthConfig.RATE_LIMIT_WINDOW,
            "burst_remaining": AuthConfig.RATE_LIMIT_BURST - len(burst_requests)
        }
    
    def _cleanup_old_requests(self, client_id: str, now: float):
        """Remove requests outside the rate limit window"""
        requests = self.requests[client_id]
        cutoff = now - AuthConfig.RATE_LIMIT_WINDOW
        
        while requests and requests[0] < cutoff:
            requests.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        now = time.time()
        stats = {
            "total_clients": len(self.requests),
            "active_clients": 0,
            "total_requests_tracked": 0,
            "configuration": {
                "requests_per_window": AuthConfig.RATE_LIMIT_REQUESTS,
                "window_seconds": AuthConfig.RATE_LIMIT_WINDOW,
                "burst_limit": AuthConfig.RATE_LIMIT_BURST
            }
        }
        
        for client_id, requests in self.requests.items():
            self._cleanup_old_requests(client_id, now)
            if requests:
                stats["active_clients"] += 1
                stats["total_requests_tracked"] += len(requests)
        
        return stats

class AuditLogger:
    """
    Audit logging for sensitive operations
    """
    
    @staticmethod
    def log_text_analysis_request(
        client_id: str,
        analysis_type: str,
        entity_type: str,
        name_hash: str,
        job_id: str,
        request_ip: str,
        user_agent: str
    ):
        """Log text analysis request for audit purposes"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "text_analysis_request",
            "client_id": client_id,
            "analysis_type": analysis_type,
            "entity_type": entity_type,
            "name_hash": name_hash,  # Hash of the name for privacy
            "job_id": job_id,
            "request_ip": request_ip,
            "user_agent": user_agent,
            "severity": "INFO"
        }
        
        audit_logger.info(f"TEXT_ANALYSIS_REQUEST: {job_id}", extra=audit_entry)
    
    @staticmethod
    def log_authentication_event(
        event_type: str,
        client_id: Optional[str],
        success: bool,
        request_ip: str,
        details: Dict[str, Any]
    ):
        """Log authentication events"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": f"auth_{event_type}",
            "client_id": client_id,
            "success": success,
            "request_ip": request_ip,
            "details": details,
            "severity": "WARNING" if not success else "INFO"
        }
        
        audit_logger.info(f"AUTH_{event_type.upper()}: {success}", extra=audit_entry)
    
    @staticmethod
    def log_rate_limit_violation(
        client_id: str,
        request_ip: str,
        limit_type: str,
        details: Dict[str, Any]
    ):
        """Log rate limit violations"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "rate_limit_violation",
            "client_id": client_id,
            "request_ip": request_ip,
            "limit_type": limit_type,
            "details": details,
            "severity": "WARNING"
        }
        
        audit_logger.warning(f"RATE_LIMIT_VIOLATION: {client_id}", extra=audit_entry)

# Global instances
rate_limiter = RateLimiter()

def get_client_id(request, credentials: Optional = None) -> str:
    """
    Generate a client ID for rate limiting and audit logging
    Uses API key if available, otherwise falls back to IP address
    """
    if credentials and credentials.credentials:
        # Use hash of API key as client ID
        return hashlib.sha256(credentials.credentials.encode()).hexdigest()[:16]
    
    # Fall back to IP address
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    return f"ip_{hashlib.sha256(client_ip.encode()).hexdigest()[:16]}"

def get_request_info(request) -> Dict[str, str]:
    """Extract request information for logging"""
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    return {
        "ip": client_ip,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "forwarded_for": forwarded_for or "none"
    }

async def authenticate_request(request, credentials: Optional = None) -> Dict[str, Any]:
    """
    Authenticate API request and apply rate limiting
    
    Returns:
        Dictionary with authentication and rate limiting info
    """
    request_info = get_request_info(request)
    client_id = get_client_id(request, credentials)
    
    # Check authentication if required
    if AuthConfig.REQUIRE_AUTH:
        if not credentials or not credentials.credentials:
            AuditLogger.log_authentication_event(
                "missing_credentials", None, False, request_info["ip"],
                {"reason": "No credentials provided"}
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication required",
                    "message": "API key must be provided in Authorization header",
                    "format": "Bearer <api_key>"
                }
            )
        
        if credentials.credentials not in AuthConfig.VALID_API_KEYS:
            AuditLogger.log_authentication_event(
                "invalid_key", client_id, False, request_info["ip"],
                {"reason": "Invalid API key"}
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        
        AuditLogger.log_authentication_event(
            "success", client_id, True, request_info["ip"],
            {"method": "api_key"}
        )
    
    # Apply rate limiting
    allowed, rate_info = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        AuditLogger.log_rate_limit_violation(
            client_id, request_info["ip"], "api_request", rate_info
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": rate_info.get("error", "Too many requests"),
                "retry_after": rate_info.get("retry_after", 3600),
                "limit": rate_info.get("limit"),
                "window": rate_info.get("window")
            },
            headers={"Retry-After": str(int(rate_info.get("retry_after", 3600)))}
        )
    
    return {
        "client_id": client_id,
        "authenticated": AuthConfig.REQUIRE_AUTH,
        "rate_limit": rate_info,
        "request_info": request_info
    }

def hash_sensitive_data(data: str) -> str:
    """
    Create a hash of sensitive data for audit logging
    Preserves privacy while allowing correlation
    """
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def log_text_analysis_audit(
    auth_info: Dict[str, Any],
    analysis_type: str,
    entity_type: str,
    name: str,
    job_id: str
):
    """
    Log text analysis request for audit purposes
    """
    if AuthConfig.AUDIT_SENSITIVE_OPERATIONS:
        AuditLogger.log_text_analysis_request(
            client_id=auth_info["client_id"],
            analysis_type=analysis_type,
            entity_type=entity_type,
            name_hash=hash_sensitive_data(name),
            job_id=job_id,
            request_ip=auth_info["request_info"]["ip"],
            user_agent=auth_info["request_info"]["user_agent"]
        )