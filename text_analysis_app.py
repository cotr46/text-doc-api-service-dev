"""
Text Analysis API Service - Character Prescreening for Debtor Candidates
Handles text-based name analysis for PEP, negative news, and law involvement screening
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import os
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any

# Google Cloud imports
from google.cloud import pubsub_v1, firestore

# Security imports
from security import InputSanitizer, SecurityViolationType, security_metrics
from auth import rate_limiter, AuthConfig, AuditLogger, get_client_id, get_request_info, hash_sensitive_data
from text_analysis_metrics import text_analysis_metrics
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize security
security = HTTPBearer(auto_error=False)

async def authenticate_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Authenticate API request and apply rate limiting"""
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

def log_text_analysis_audit(
    auth_info: Dict[str, Any],
    analysis_type: str,
    entity_type: str,
    name: str,
    job_id: str
):
    """Log text analysis request for audit purposes"""
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

# Text analysis enums
class AnalysisType(str, Enum):
    PEP_ANALYSIS = "pep-analysis"
    NEGATIVE_NEWS = "negative-news"
    LAW_INVOLVEMENT = "law-involvement"
    CORPORATE_NEGATIVE_NEWS = "corporate-negative-news"
    CORPORATE_LAW_INVOLVEMENT = "corporate-law-involvement"

class EntityType(str, Enum):
    PERSON = "person"
    CORPORATE = "corporate"

# Job status enum
class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Text analysis model configuration
TEXT_MODEL_CONFIG = {
    "pep-analysis": {
        "model": "politically-exposed-person-v2",
        "entity_types": ["person"],
        "description": "Political Exposure Person Analysis v2"
    },
    "negative-news": {
        "model": "negative-news", 
        "entity_types": ["person"],
        "description": "Individual Negative News Analysis"
    },
    "law-involvement": {
        "model": "law-involvement",
        "entity_types": ["person"], 
        "description": "Individual Law Involvement Analysis"
    },
    "corporate-negative-news": {
        "model": "negative-news-corporate",
        "entity_types": ["corporate"],
        "description": "Corporate Negative News Analysis"
    },
    "corporate-law-involvement": {
        "model": "law-involvement-corporate",
        "entity_types": ["corporate"],
        "description": "Corporate Law Involvement Analysis"
    }
}

# Pydantic models for text analysis
class TextAnalysisRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Name to analyze (person or corporate)")
    entity_type: EntityType = Field(..., description="Type of entity being analyzed")
    additional_context: Optional[str] = Field(None, max_length=500, description="Optional additional context")
    
    @validator('name')
    def validate_name(cls, v):
        """Enhanced name validation with comprehensive security checks"""
        try:
            sanitized_name = InputSanitizer.sanitize_name(v, "name")
            security_metrics.record_validation(True)
            return sanitized_name
        except ValueError as e:
            security_metrics.record_validation(False, SecurityViolationType.INVALID_CHARACTERS)
            raise e
    
    @validator('additional_context')
    def validate_context(cls, v):
        """Enhanced context validation with security checks"""
        if v is None:
            return v
        try:
            sanitized_context = InputSanitizer.sanitize_context(v)
            security_metrics.record_validation(True)
            return sanitized_context
        except ValueError as e:
            security_metrics.record_validation(False, SecurityViolationType.INVALID_CHARACTERS)
            raise e

class TextAnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Allow model_* field names
    
    success: bool
    job_id: str
    status: str
    analysis_type: str
    entity_type: str
    name: str
    model_name: str
    submitted_at: str
    processing_time: float
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Text Analysis API - Character Prescreening",
    description="Character prescreening service for debtor candidates. Analyzes person and corporate names for PEP screening, negative news, and law involvement using custom search API.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for GCP clients (initialized at startup)
publisher = None
firestore_client = None

# Configuration constants
class ServiceConfig:
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bni-prod-dma-bnimove-ai")
    PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "text-analysis-request")
    FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "text-analysis-firestore")

# Environment variables
PROJECT_ID = ServiceConfig.PROJECT_ID
PUBSUB_TOPIC = ServiceConfig.PUBSUB_TOPIC
FIRESTORE_DATABASE = ServiceConfig.FIRESTORE_DATABASE

def initialize_gcp_clients():
    """Initialize Google Cloud clients"""
    global publisher, firestore_client
    
    try:
        # Initialize Pub/Sub publisher
        publisher = pubsub_v1.PublisherClient()
        
        # Initialize Firestore client
        firestore_client = firestore.Client(
            project=PROJECT_ID, 
            database=FIRESTORE_DATABASE
        )
        
        print(f"✅ GCP clients initialized for project: {PROJECT_ID}")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize GCP clients: {e}")
        print(f"⚠️ Running in LOCAL MODE - GCP operations will fail")
        print(f"⚠️ To fix: Run 'gcloud auth application-default login'")

@app.on_event("startup")
async def startup_event():
    """Initialize GCP clients on startup"""
    initialize_gcp_clients()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Text Analysis API - Character Prescreening",
        "status": "running",
        "version": "1.0.0",
        "purpose": "Debtor candidate character prescreening",
        "available_analyses": list(TEXT_MODEL_CONFIG.keys()),
        "features": {
            "pep_screening": "Political Exposure Person analysis",
            "negative_news": "Person and corporate negative news screening",
            "law_involvement": "Person and corporate law involvement check",
            "custom_search_api": "Powered by custom search API"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "infrastructure": {
            "project_id": PROJECT_ID,
            "pubsub_topic": PUBSUB_TOPIC,
            "firestore_db": FIRESTORE_DATABASE
        }
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    health_status = {"status": "healthy", "checks": {}}
    
    if firestore_client is None or publisher is None:
        health_status["status"] = "degraded"
        health_status["checks"]["gcp_clients"] = "❌ not initialized (local mode)"
        health_status["mode"] = "local"
        return health_status
    
    try:
        firestore_client.collection("jobs").limit(1).get()
        health_status["checks"]["firestore"] = "✅ accessible"
    except Exception as e:
        health_status["checks"]["firestore"] = f"❌ error: {str(e)[:100]}"
        health_status["status"] = "degraded"
    
    try:
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        publisher.get_topic(request={"topic": topic_path})
        health_status["checks"]["pubsub"] = "✅ accessible"
    except Exception as e:
        health_status["checks"]["pubsub"] = f"❌ error: {str(e)[:100]}"
        health_status["status"] = "degraded"
    
    health_status["service_capabilities"] = {
        "text_analysis": "✅ enabled",
        "authentication": "✅ enabled" if AuthConfig.REQUIRE_AUTH else "⚠️ disabled",
        "rate_limiting": "✅ enabled",
        "security_validation": "✅ enabled",
        "metrics_tracking": "✅ enabled",
        "audit_logging": "✅ enabled" if AuthConfig.AUDIT_SENSITIVE_OPERATIONS else "⚠️ disabled"
    }
    
    try:
        text_metrics = text_analysis_metrics.get_comprehensive_metrics()
        health_status["metrics_summary"] = {
            "total_requests": text_metrics["overview"]["total_requests"],
            "success_rate": text_metrics["overview"]["success_rate"],
            "avg_latency_ms": text_metrics["overview"]["avg_latency_ms"],
            "requests_per_minute": text_metrics["overview"]["requests_per_minute"]
        }
    except Exception:
        pass
    
    return health_status


def create_text_analysis_job_record(job_id: str, analysis_type: str, entity_type: str, 
                                  name: str, additional_context: Optional[str] = None) -> dict:
    """Create text analysis job record in Firestore"""
    try:
        model_config = TEXT_MODEL_CONFIG[analysis_type]
        
        job_data = {
            "job_id": job_id,
            "job_type": "text_analysis",
            "status": JobStatus.SUBMITTED.value,
            "analysis_type": analysis_type,
            "entity_type": entity_type,
            "name": name,
            "additional_context": additional_context,
            "model_name": model_config["model"],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "result": None,
            "error": None
        }
        
        doc_ref = firestore_client.collection("jobs").document(job_id)
        doc_ref.set(job_data)
        
        print(f"✅ Text analysis job record created: {job_id} ({analysis_type} for {entity_type})")
        return job_data
        
    except Exception as e:
        print(f"❌ Firestore text analysis job creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis job creation failed: {str(e)}")

def publish_text_analysis_message(job_id: str, analysis_type: str, entity_type: str, 
                                name: str, additional_context: Optional[str] = None):
    """Publish text analysis message to Pub/Sub for processing"""
    try:
        model_config = TEXT_MODEL_CONFIG[analysis_type]
        
        message_data = {
            "job_id": job_id,
            "job_type": "text_analysis",
            "analysis_type": analysis_type,
            "entity_type": entity_type,
            "name": name,
            "additional_context": additional_context,
            "model_name": model_config["model"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"🔍 Publishing text analysis message:")
        print(f"   - Job ID: {job_id}")
        print(f"   - Analysis type: {analysis_type}")
        print(f"   - Entity type: {entity_type}")
        print(f"   - Model: {model_config['model']}")
        
        message_json = json.dumps(message_data).encode('utf-8')
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        future = publisher.publish(topic_path, message_json)
        message_id = future.result()
        
        print(f"✅ Text analysis message published to Pub/Sub: {message_id}")
        return message_id
        
    except Exception as e:
        print(f"❌ Text analysis Pub/Sub publish failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis message publish failed: {str(e)}")

@app.post("/analyze/{analysis_type}", response_model=TextAnalysisResponse)
async def analyze_text(
    analysis_type: AnalysisType,
    request_data: TextAnalysisRequest,
    auth_info: Dict[str, Any] = Depends(authenticate_request)
):
    """
    Perform character prescreening analysis on debtor candidate
    
    - **analysis_type**: Type of analysis (pep-analysis, negative-news, law-involvement, etc.)
    - **name**: Name of person or corporate to analyze
    - **entity_type**: Type of entity (person or corporate)
    - **additional_context**: Optional additional context for analysis
    """
    start_time = time.time()
    
    try:
        if analysis_type not in TEXT_MODEL_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported analysis type '{analysis_type}'"
            )
        
        model_config = TEXT_MODEL_CONFIG[analysis_type]
        if request_data.entity_type.value not in model_config["entity_types"]:
            raise HTTPException(
                status_code=400,
                detail=f"Entity type '{request_data.entity_type}' not supported for {analysis_type}"
            )
        
        job_id = f"text-{uuid.uuid4()}"
        
        log_text_analysis_audit(
            auth_info,
            analysis_type.value,
            request_data.entity_type.value,
            request_data.name,
            job_id
        )
        
        create_text_analysis_job_record(
            job_id,
            analysis_type.value,
            request_data.entity_type.value,
            request_data.name,
            request_data.additional_context
        )
        
        publish_text_analysis_message(
            job_id,
            analysis_type.value,
            request_data.entity_type.value,
            request_data.name,
            request_data.additional_context
        )
        
        processing_time = time.time() - start_time
        
        text_analysis_metrics.record_request(
            analysis_type.value,
            request_data.entity_type.value,
            True,
            processing_time
        )
        
        return TextAnalysisResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.SUBMITTED.value,
            analysis_type=analysis_type.value,
            entity_type=request_data.entity_type.value,
            name=request_data.name,
            model_name=model_config["model"],
            submitted_at=datetime.now(timezone.utc).isoformat(),
            processing_time=round(processing_time, 3),
            message=f"Character prescreening job submitted successfully. Job ID: {job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        text_analysis_metrics.record_request(
            analysis_type.value,
            request_data.entity_type.value,
            False,
            processing_time
        )
        
        print(f"❌ Text analysis request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis request failed: {str(e)}"
        )

@app.get("/job/{job_id}")
async def get_job_status(
    job_id: str,
    auth_info: Dict[str, Any] = Depends(authenticate_request)
):
    """Get status and result of a text analysis job"""
    try:
        doc_ref = firestore_client.collection("jobs").document(job_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_data = doc.to_dict()
        
        if job_data.get("created_at"):
            job_data["created_at"] = job_data["created_at"].isoformat()
        if job_data.get("updated_at"):
            job_data["updated_at"] = job_data["updated_at"].isoformat()
        if job_data.get("completed_at"):
            job_data["completed_at"] = job_data["completed_at"].isoformat()
        
        return {
            "success": True,
            "job": job_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/metrics")
async def get_metrics(auth_info: Dict[str, Any] = Depends(authenticate_request)):
    """Get comprehensive metrics for text analysis service"""
    try:
        metrics = text_analysis_metrics.get_comprehensive_metrics()
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"❌ Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
