"""
Model Armor Client for AI Security
Screens prompts and responses for security risks
"""

import os
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FilterMatchState(str, Enum):
    """Match state from Model Armor"""
    MATCH_FOUND = "MATCH_FOUND"
    NO_MATCH_FOUND = "NO_MATCH_FOUND"
    MATCH_STATE_UNSPECIFIED = "MATCH_STATE_UNSPECIFIED"


@dataclass
class SanitizeResult:
    """Result from Model Armor sanitization"""
    blocked: bool
    match_state: str
    prompt_injection_detected: bool
    jailbreak_detected: bool
    malicious_uri_detected: bool
    sensitive_data_detected: bool
    sanitized_content: Optional[str]
    raw_response: Dict[str, Any]
    latency_ms: float
    error: Optional[str] = None


class ModelArmorClient:
    """
    Client for Google Cloud Model Armor API
    Provides prompt and response sanitization for AI security
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Model Armor client"""
        config = config or {}
        
        # Project ID from Secret Manager (required)
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        
        self.location = config.get("location", os.getenv("MODEL_ARMOR_LOCATION", "asia-southeast1"))
        self.template_id = config.get("template_id", os.getenv("MODEL_ARMOR_TEMPLATE_ID", "text-analysis-security"))
        # Disabled by default until asia-southeast2 is supported or org policy exception granted
        self.enabled = config.get("enabled", os.getenv("MODEL_ARMOR_ENABLED", "false").lower() == "true")
        self.timeout = config.get("timeout", 10)  # 10 seconds timeout
        
        # Template path
        self.template_path = f"projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}"
        
        # API endpoint
        self.base_url = f"https://modelarmor.googleapis.com/v1/{self.template_path}"
        
        # Get access token (will be refreshed as needed)
        self._access_token = None
        self._token_expiry = 0
        
        self._log(f"ModelArmorClient initialized")
        self._log(f"  Template: {self.template_path}")
        self._log(f"  Enabled: {self.enabled}")
    
    def _log(self, message: str) -> None:
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ModelArmor: {message}", flush=True)
    
    def _get_access_token(self) -> str:
        """Get or refresh access token"""
        import google.auth
        import google.auth.transport.requests
        
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token
        
        credentials, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        
        self._access_token = credentials.token
        self._token_expiry = time.time() + 3600  # Assume 1 hour validity
        
        return self._access_token
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Model Armor API"""
        url = f"{self.base_url}:{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Model Armor API error: {response.status_code} - {response.text[:200]}")
        
        return response.json()
    
    def sanitize_prompt(self, user_input: str) -> SanitizeResult:
        """
        Sanitize user prompt before sending to AI model
        
        Args:
            user_input: The user's input text to check
            
        Returns:
            SanitizeResult with detection results
        """
        if not self.enabled:
            return SanitizeResult(
                blocked=False,
                match_state=FilterMatchState.NO_MATCH_FOUND.value,
                prompt_injection_detected=False,
                jailbreak_detected=False,
                malicious_uri_detected=False,
                sensitive_data_detected=False,
                sanitized_content=user_input,
                raw_response={},
                latency_ms=0,
                error=None
            )
        
        start_time = time.time()
        
        try:
            payload = {
                "userPromptData": {
                    "text": user_input
                }
            }
            
            response = self._make_request("sanitizeUserPrompt", payload)
            latency_ms = (time.time() - start_time) * 1000
            
            return self._parse_sanitize_response(response, user_input, latency_ms)
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._log(f"❌ Sanitize prompt error: {str(e)}")
            
            # On error, allow request but log
            return SanitizeResult(
                blocked=False,
                match_state=FilterMatchState.MATCH_STATE_UNSPECIFIED.value,
                prompt_injection_detected=False,
                jailbreak_detected=False,
                malicious_uri_detected=False,
                sensitive_data_detected=False,
                sanitized_content=user_input,
                raw_response={},
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def sanitize_response(self, model_output: str) -> SanitizeResult:
        """
        Sanitize AI model response before returning to user
        
        Args:
            model_output: The AI model's response text to check
            
        Returns:
            SanitizeResult with detection results
        """
        if not self.enabled:
            return SanitizeResult(
                blocked=False,
                match_state=FilterMatchState.NO_MATCH_FOUND.value,
                prompt_injection_detected=False,
                jailbreak_detected=False,
                malicious_uri_detected=False,
                sensitive_data_detected=False,
                sanitized_content=model_output,
                raw_response={},
                latency_ms=0,
                error=None
            )
        
        start_time = time.time()
        
        try:
            payload = {
                "modelResponseData": {
                    "text": model_output
                }
            }
            
            response = self._make_request("sanitizeModelResponse", payload)
            latency_ms = (time.time() - start_time) * 1000
            
            return self._parse_sanitize_response(response, model_output, latency_ms)
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._log(f"❌ Sanitize response error: {str(e)}")
            
            # On error, allow response but log
            return SanitizeResult(
                blocked=False,
                match_state=FilterMatchState.MATCH_STATE_UNSPECIFIED.value,
                prompt_injection_detected=False,
                jailbreak_detected=False,
                malicious_uri_detected=False,
                sensitive_data_detected=False,
                sanitized_content=model_output,
                raw_response={},
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _parse_sanitize_response(self, response: Dict[str, Any], original_text: str, latency_ms: float) -> SanitizeResult:
        """Parse Model Armor API response"""
        
        # Get sanitization result
        sanitization_result = response.get("sanitizationResult", {})
        filter_match_state = sanitization_result.get("filterMatchState", "MATCH_STATE_UNSPECIFIED")
        filter_results = sanitization_result.get("filterResults", {})
        
        # Check individual filters
        pi_jailbreak = filter_results.get("piAndJailbreakFilterResult", {})
        pi_detected = pi_jailbreak.get("promptInjectionResult", {}).get("matchState") == "MATCH_FOUND"
        jailbreak_detected = pi_jailbreak.get("jailbreakResult", {}).get("matchState") == "MATCH_FOUND"
        
        malicious_uri = filter_results.get("maliciousUriFilterResult", {})
        malicious_uri_detected = malicious_uri.get("matchState") == "MATCH_FOUND"
        
        sdp_result = filter_results.get("sdpFilterResult", {})
        sensitive_data_detected = sdp_result.get("inspectResult", {}).get("matchState") == "MATCH_FOUND"
        
        # Determine if blocked
        blocked = filter_match_state == "MATCH_FOUND"
        
        # Get sanitized content if available
        sanitized_content = original_text
        if "sanitizedModelResponseData" in response:
            sanitized_content = response["sanitizedModelResponseData"].get("text", original_text)
        
        if blocked:
            self._log(f"⚠️ Content blocked - PI:{pi_detected} JB:{jailbreak_detected} URI:{malicious_uri_detected} PII:{sensitive_data_detected}")
        else:
            self._log(f"✅ Content passed ({latency_ms:.0f}ms)")
        
        return SanitizeResult(
            blocked=blocked,
            match_state=filter_match_state,
            prompt_injection_detected=pi_detected,
            jailbreak_detected=jailbreak_detected,
            malicious_uri_detected=malicious_uri_detected,
            sensitive_data_detected=sensitive_data_detected,
            sanitized_content=sanitized_content,
            raw_response=response,
            latency_ms=latency_ms,
            error=None
        )


# Singleton instance
_model_armor_client: Optional[ModelArmorClient] = None


def get_model_armor_client() -> ModelArmorClient:
    """Get or create Model Armor client singleton"""
    global _model_armor_client
    if _model_armor_client is None:
        _model_armor_client = ModelArmorClient()
    return _model_armor_client


async def sanitize_user_prompt(user_input: str) -> SanitizeResult:
    """Async wrapper for sanitize_prompt"""
    client = get_model_armor_client()
    return client.sanitize_prompt(user_input)


async def sanitize_model_response(model_output: str) -> SanitizeResult:
    """Async wrapper for sanitize_response"""
    client = get_model_armor_client()
    return client.sanitize_response(model_output)
