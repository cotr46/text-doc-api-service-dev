"""
Security and Input Sanitization Module for Text Analysis
Provides comprehensive input validation, sanitization, and security logging
"""

import re
import logging
import html
import unicodedata
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

# Configure security logger
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)

# Create handler if not exists
if not security_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)

class SecurityViolationType(str, Enum):
    """Types of security violations for logging"""
    INVALID_CHARACTERS = "invalid_characters"
    EXCESSIVE_LENGTH = "excessive_length"
    INJECTION_ATTEMPT = "injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    EMPTY_INPUT = "empty_input"
    ENCODING_ISSUE = "encoding_issue"

class InputSanitizer:
    """
    Comprehensive input sanitization for text analysis
    Handles name validation, character filtering, and security logging
    """
    
    # Character patterns for validation
    ALLOWED_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9\s\.\-\,\'\&\(\)\u00C0-\u017F\u0100-\u024F]+$')
    SUSPICIOUS_PATTERNS = [
        re.compile(r'<[^>]*>', re.IGNORECASE),  # HTML tags
        re.compile(r'javascript:', re.IGNORECASE),  # JavaScript
        re.compile(r'data:', re.IGNORECASE),  # Data URLs
        re.compile(r'vbscript:', re.IGNORECASE),  # VBScript
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s+', re.IGNORECASE),  # SQL
        re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),  # Control characters
        re.compile(r'[{}[\]\\]'),  # Potentially dangerous brackets
    ]
    
    # Length constraints
    MIN_NAME_LENGTH = 1
    MAX_NAME_LENGTH = 200
    MAX_CONTEXT_LENGTH = 500
    
    @classmethod
    def sanitize_name(cls, name: str, field_name: str = "name") -> str:
        """
        Sanitize and validate name input with comprehensive security checks
        
        Args:
            name: Input name to sanitize
            field_name: Name of the field for logging purposes
            
        Returns:
            Sanitized name string
            
        Raises:
            ValueError: If input fails validation or contains security violations
        """
        if not isinstance(name, str):
            cls._log_security_violation(
                SecurityViolationType.INVALID_CHARACTERS,
                f"Non-string input for {field_name}",
                {"input_type": type(name).__name__}
            )
            raise ValueError(f"{field_name} must be a string")
        
        # Check for empty input
        if not name or not name.strip():
            cls._log_security_violation(
                SecurityViolationType.EMPTY_INPUT,
                f"Empty {field_name} submitted",
                {"original_input": repr(name)}
            )
            raise ValueError(f'{field_name} cannot be empty or whitespace only')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', name.strip())
        
        # Check length constraints
        if len(sanitized) < cls.MIN_NAME_LENGTH:
            cls._log_security_violation(
                SecurityViolationType.EXCESSIVE_LENGTH,
                f"{field_name} too short",
                {"length": len(sanitized), "min_length": cls.MIN_NAME_LENGTH}
            )
            raise ValueError(f'{field_name} must be at least {cls.MIN_NAME_LENGTH} character(s)')
        
        if len(sanitized) > cls.MAX_NAME_LENGTH:
            cls._log_security_violation(
                SecurityViolationType.EXCESSIVE_LENGTH,
                f"{field_name} too long",
                {"length": len(sanitized), "max_length": cls.MAX_NAME_LENGTH}
            )
            raise ValueError(f'{field_name} must not exceed {cls.MAX_NAME_LENGTH} characters')
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if pattern.search(sanitized):
                cls._log_security_violation(
                    SecurityViolationType.INJECTION_ATTEMPT,
                    f"Suspicious pattern detected in {field_name}",
                    {"pattern": pattern.pattern, "input": sanitized[:100]}
                )
                raise ValueError(f'{field_name} contains invalid characters or patterns')
        
        # Validate allowed characters (including international characters)
        if not cls.ALLOWED_NAME_PATTERN.match(sanitized):
            cls._log_security_violation(
                SecurityViolationType.INVALID_CHARACTERS,
                f"Invalid characters in {field_name}",
                {"input": sanitized[:100]}
            )
            raise ValueError(
                f'{field_name} contains invalid characters. Only letters, numbers, spaces, '
                'and basic punctuation (.-,\'&()) are allowed'
            )
        
        # Additional Unicode normalization for security
        try:
            sanitized = unicodedata.normalize('NFKC', sanitized)
        except Exception as e:
            cls._log_security_violation(
                SecurityViolationType.ENCODING_ISSUE,
                f"Unicode normalization failed for {field_name}",
                {"error": str(e), "input": sanitized[:100]}
            )
            raise ValueError(f'{field_name} contains invalid encoding')
        
        # HTML escape as additional protection
        sanitized = html.escape(sanitized, quote=False)
        
        return sanitized
    
    @classmethod
    def sanitize_context(cls, context: Optional[str]) -> Optional[str]:
        """
        Sanitize additional context field with similar security checks
        
        Args:
            context: Optional context string
            
        Returns:
            Sanitized context string or None
        """
        if context is None:
            return None
        
        if not isinstance(context, str):
            cls._log_security_violation(
                SecurityViolationType.INVALID_CHARACTERS,
                "Non-string input for additional_context",
                {"input_type": type(context).__name__}
            )
            raise ValueError("additional_context must be a string")
        
        # Normalize whitespace
        sanitized = context.strip()
        if not sanitized:
            return None
        
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Check length
        if len(sanitized) > cls.MAX_CONTEXT_LENGTH:
            cls._log_security_violation(
                SecurityViolationType.EXCESSIVE_LENGTH,
                "additional_context too long",
                {"length": len(sanitized), "max_length": cls.MAX_CONTEXT_LENGTH}
            )
            raise ValueError(f'additional_context must not exceed {cls.MAX_CONTEXT_LENGTH} characters')
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if pattern.search(sanitized):
                cls._log_security_violation(
                    SecurityViolationType.INJECTION_ATTEMPT,
                    "Suspicious pattern detected in additional_context",
                    {"pattern": pattern.pattern, "input": sanitized[:100]}
                )
                raise ValueError('additional_context contains invalid characters or patterns')
        
        # Unicode normalization
        try:
            sanitized = unicodedata.normalize('NFKC', sanitized)
        except Exception as e:
            cls._log_security_violation(
                SecurityViolationType.ENCODING_ISSUE,
                "Unicode normalization failed for additional_context",
                {"error": str(e), "input": sanitized[:100]}
            )
            raise ValueError('additional_context contains invalid encoding')
        
        # HTML escape
        sanitized = html.escape(sanitized, quote=False)
        
        return sanitized
    
    @classmethod
    def validate_analysis_parameters(cls, analysis_type: str, entity_type: str, name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of all text analysis parameters
        
        Args:
            analysis_type: Type of analysis requested
            entity_type: Type of entity (person/corporate)
            name: Name to analyze
            
        Returns:
            Dictionary with validation results and sanitized values
        """
        validation_result = {
            "valid": True,
            "sanitized_name": None,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Sanitize name
            validation_result["sanitized_name"] = cls.sanitize_name(name)
            
            # Additional business logic validation
            if entity_type == "person" and len(name.split()) > 10:
                validation_result["warnings"].append("Person name has unusually many parts")
            
            if entity_type == "corporate" and len(name) < 3:
                validation_result["warnings"].append("Corporate name is unusually short")
            
        except ValueError as e:
            validation_result["valid"] = False
            validation_result["violations"].append(str(e))
        
        return validation_result
    
    @classmethod
    def _log_security_violation(cls, violation_type: SecurityViolationType, 
                              message: str, details: Dict[str, Any]):
        """
        Log security violations for monitoring and audit purposes
        
        Args:
            violation_type: Type of security violation
            message: Human-readable message
            details: Additional details for investigation
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "violation_type": violation_type.value,
            "violation_message": message,  # Changed from 'message' to avoid conflict
            "details": details,
            "severity": "HIGH" if violation_type in [
                SecurityViolationType.INJECTION_ATTEMPT,
                SecurityViolationType.SUSPICIOUS_PATTERN
            ] else "MEDIUM"
        }
        
        security_logger.warning(f"SECURITY_VIOLATION: {message}", extra=log_entry)

class SecurityMetrics:
    """
    Track security-related metrics for monitoring
    """
    
    def __init__(self):
        self.violation_counts = {}
        self.total_validations = 0
        self.failed_validations = 0
    
    def record_validation(self, success: bool, violation_type: Optional[SecurityViolationType] = None):
        """Record a validation attempt"""
        self.total_validations += 1
        
        if not success:
            self.failed_validations += 1
            
            if violation_type:
                if violation_type not in self.violation_counts:
                    self.violation_counts[violation_type] = 0
                self.violation_counts[violation_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        return {
            "total_validations": self.total_validations,
            "failed_validations": self.failed_validations,
            "success_rate": (self.total_validations - self.failed_validations) / max(self.total_validations, 1),
            "violation_counts": dict(self.violation_counts),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

# Global security metrics instance
security_metrics = SecurityMetrics()