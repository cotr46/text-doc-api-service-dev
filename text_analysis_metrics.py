"""
Text Analysis Metrics Module
Tracks request volume, latency, success/failure rates, and model performance
Requirements: 8.1, 10.1, 10.3
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from enum import Enum


class AnalysisType(str, Enum):
    """Analysis types for metrics tracking"""
    PEP_ANALYSIS = "pep-analysis"
    NEGATIVE_NEWS = "negative-news"
    LAW_INVOLVEMENT = "law-involvement"
    CORPORATE_NEGATIVE_NEWS = "corporate-negative-news"
    CORPORATE_LAW_INVOLVEMENT = "corporate-law-involvement"


class EntityType(str, Enum):
    """Entity types for metrics tracking"""
    PERSON = "person"
    CORPORATE = "corporate"


class TextAnalysisMetrics:
    """
    Comprehensive metrics tracking for text analysis operations
    Thread-safe implementation for concurrent access
    """
    
    def __init__(self, max_latency_samples: int = 1000):
        """
        Initialize metrics tracking
        
        Args:
            max_latency_samples: Maximum number of latency samples to keep in memory
        """
        self._lock = threading.RLock()
        self.max_latency_samples = max_latency_samples
        
        # Request volume metrics
        self.total_requests = 0
        self.requests_by_analysis_type = defaultdict(int)
        self.requests_by_entity_type = defaultdict(int)
        self.requests_by_model = defaultdict(int)
        
        # Success/failure tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.failures_by_analysis_type = defaultdict(int)
        self.failures_by_entity_type = defaultdict(int)
        self.failures_by_model = defaultdict(int)
        self.failure_reasons = defaultdict(int)
        
        # Latency tracking (using deque for efficient memory management)
        self.request_latencies = deque(maxlen=max_latency_samples)
        self.latencies_by_analysis_type = defaultdict(lambda: deque(maxlen=max_latency_samples))
        self.latencies_by_model = defaultdict(lambda: deque(maxlen=max_latency_samples))
        
        # Model response times and availability
        self.model_response_times = defaultdict(lambda: deque(maxlen=max_latency_samples))
        self.model_availability_checks = defaultdict(int)
        self.model_unavailable_count = defaultdict(int)
        self.model_timeout_count = defaultdict(int)
        
        # Time-based metrics
        self.start_time = datetime.now(timezone.utc)
        self.last_request_time = None
        self.requests_per_minute = deque(maxlen=60)  # Track requests per minute for last hour
        
        # Fallback usage tracking
        self.fallback_usage_count = defaultdict(int)
        self.primary_model_failures = defaultdict(int)
        
        print("📊 TextAnalysisMetrics initialized with comprehensive tracking")
    
    def record_request_start(self, analysis_type: str, entity_type: str, model_name: str) -> float:
        """
        Record the start of a text analysis request
        
        Args:
            analysis_type: Type of analysis being performed
            entity_type: Type of entity being analyzed
            model_name: Name of the model being used
            
        Returns:
            Start timestamp for latency calculation
        """
        start_time = time.time()
        
        with self._lock:
            self.total_requests += 1
            self.requests_by_analysis_type[analysis_type] += 1
            self.requests_by_entity_type[entity_type] += 1
            self.requests_by_model[model_name] += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            # Track requests per minute
            current_minute = int(time.time() // 60)
            self.requests_per_minute.append(current_minute)
        
        return start_time
    
    def record_request_success(self, start_time: float, analysis_type: str, 
                             entity_type: str, model_name: str, 
                             model_response_time: Optional[float] = None):
        """
        Record a successful text analysis request
        
        Args:
            start_time: Start timestamp from record_request_start
            analysis_type: Type of analysis performed
            entity_type: Type of entity analyzed
            model_name: Name of the model used
            model_response_time: Time taken by the model to respond (optional)
        """
        end_time = time.time()
        total_latency = end_time - start_time
        
        with self._lock:
            self.successful_requests += 1
            
            # Record latencies
            self.request_latencies.append(total_latency)
            self.latencies_by_analysis_type[analysis_type].append(total_latency)
            self.latencies_by_model[model_name].append(total_latency)
            
            # Record model response time if provided
            if model_response_time is not None:
                self.model_response_times[model_name].append(model_response_time)
    
    def record_request_failure(self, start_time: float, analysis_type: str, 
                             entity_type: str, model_name: str, 
                             failure_reason: str):
        """
        Record a failed text analysis request
        
        Args:
            start_time: Start timestamp from record_request_start
            analysis_type: Type of analysis attempted
            entity_type: Type of entity being analyzed
            model_name: Name of the model attempted
            failure_reason: Reason for failure
        """
        end_time = time.time()
        total_latency = end_time - start_time
        
        with self._lock:
            self.failed_requests += 1
            self.failures_by_analysis_type[analysis_type] += 1
            self.failures_by_entity_type[entity_type] += 1
            self.failures_by_model[model_name] += 1
            self.failure_reasons[failure_reason] += 1
            
            # Still record latency for failed requests (useful for timeout analysis)
            self.request_latencies.append(total_latency)
    
    def record_model_availability_check(self, model_name: str, is_available: bool, 
                                      response_time: Optional[float] = None):
        """
        Record a model availability check
        
        Args:
            model_name: Name of the model checked
            is_available: Whether the model was available
            response_time: Time taken for the availability check
        """
        with self._lock:
            self.model_availability_checks[model_name] += 1
            
            if not is_available:
                self.model_unavailable_count[model_name] += 1
            
            if response_time is not None:
                if response_time > 30:  # Consider >30s as timeout
                    self.model_timeout_count[model_name] += 1
    
    def record_fallback_usage(self, original_analysis_type: str, fallback_analysis_type: str):
        """
        Record usage of fallback model
        
        Args:
            original_analysis_type: Original analysis type that failed
            fallback_analysis_type: Fallback analysis type used
        """
        with self._lock:
            self.fallback_usage_count[f"{original_analysis_type}->{fallback_analysis_type}"] += 1
            self.primary_model_failures[original_analysis_type] += 1
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for monitoring and observability
        
        Returns:
            Dictionary containing all tracked metrics
        """
        with self._lock:
            current_time = datetime.now(timezone.utc)
            uptime_seconds = (current_time - self.start_time).total_seconds()
            
            # Calculate success rate
            total_completed = self.successful_requests + self.failed_requests
            success_rate = (self.successful_requests / max(total_completed, 1)) * 100
            
            # Calculate average latencies
            avg_latency = sum(self.request_latencies) / max(len(self.request_latencies), 1)
            
            # Calculate requests per minute (last 5 minutes)
            current_minute = int(time.time() // 60)
            recent_requests = sum(1 for minute in self.requests_per_minute 
                                if current_minute - minute <= 5)
            requests_per_minute = recent_requests / 5.0
            
            # Model availability rates
            model_availability = {}
            for model_name in self.model_availability_checks:
                total_checks = self.model_availability_checks[model_name]
                unavailable = self.model_unavailable_count[model_name]
                availability_rate = ((total_checks - unavailable) / max(total_checks, 1)) * 100
                
                avg_response_time = None
                if model_name in self.model_response_times and self.model_response_times[model_name]:
                    avg_response_time = sum(self.model_response_times[model_name]) / len(self.model_response_times[model_name])
                
                model_availability[model_name] = {
                    "availability_rate": round(availability_rate, 2),
                    "total_checks": total_checks,
                    "unavailable_count": unavailable,
                    "timeout_count": self.model_timeout_count[model_name],
                    "avg_response_time_ms": round(avg_response_time * 1000, 2) if avg_response_time else None
                }
            
            # Analysis type performance
            analysis_type_metrics = {}
            for analysis_type in self.requests_by_analysis_type:
                total_requests = self.requests_by_analysis_type[analysis_type]
                failures = self.failures_by_analysis_type[analysis_type]
                success_rate_type = ((total_requests - failures) / max(total_requests, 1)) * 100
                
                avg_latency_type = None
                if analysis_type in self.latencies_by_analysis_type and self.latencies_by_analysis_type[analysis_type]:
                    avg_latency_type = sum(self.latencies_by_analysis_type[analysis_type]) / len(self.latencies_by_analysis_type[analysis_type])
                
                analysis_type_metrics[analysis_type] = {
                    "total_requests": total_requests,
                    "failures": failures,
                    "success_rate": round(success_rate_type, 2),
                    "avg_latency_ms": round(avg_latency_type * 1000, 2) if avg_latency_type else None
                }
            
            return {
                "overview": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "success_rate": round(success_rate, 2),
                    "avg_latency_ms": round(avg_latency * 1000, 2),
                    "requests_per_minute": round(requests_per_minute, 2),
                    "uptime_seconds": round(uptime_seconds, 2),
                    "last_request": self.last_request_time.isoformat() if self.last_request_time else None
                },
                "by_analysis_type": dict(analysis_type_metrics),
                "by_entity_type": {
                    "requests": dict(self.requests_by_entity_type),
                    "failures": dict(self.failures_by_entity_type)
                },
                "by_model": {
                    "requests": dict(self.requests_by_model),
                    "failures": dict(self.failures_by_model),
                    "availability": model_availability
                },
                "failure_analysis": {
                    "failure_reasons": dict(self.failure_reasons),
                    "fallback_usage": dict(self.fallback_usage_count),
                    "primary_model_failures": dict(self.primary_model_failures)
                },
                "performance": {
                    "latency_percentiles": self._calculate_latency_percentiles(),
                    "model_response_times": self._get_model_response_time_stats()
                },
                "metadata": {
                    "metrics_collected_at": current_time.isoformat(),
                    "metrics_start_time": self.start_time.isoformat(),
                    "sample_sizes": {
                        "total_latency_samples": len(self.request_latencies),
                        "max_samples_per_metric": self.max_latency_samples
                    }
                }
            }
    
    def _calculate_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles from collected samples"""
        if not self.request_latencies:
            return {}
        
        sorted_latencies = sorted(self.request_latencies)
        n = len(sorted_latencies)
        
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            index = int((p / 100) * n) - 1
            if index < 0:
                index = 0
            elif index >= n:
                index = n - 1
            percentiles[f"p{p}"] = round(sorted_latencies[index] * 1000, 2)  # Convert to ms
        
        return percentiles
    
    def _get_model_response_time_stats(self) -> Dict[str, Dict[str, float]]:
        """Get response time statistics for each model"""
        stats = {}
        
        for model_name, response_times in self.model_response_times.items():
            if response_times:
                sorted_times = sorted(response_times)
                n = len(sorted_times)
                
                stats[model_name] = {
                    "avg_ms": round(sum(sorted_times) / n * 1000, 2),
                    "min_ms": round(min(sorted_times) * 1000, 2),
                    "max_ms": round(max(sorted_times) * 1000, 2),
                    "median_ms": round(sorted_times[n // 2] * 1000, 2),
                    "sample_count": n
                }
        
        return stats
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing or periodic resets)"""
        with self._lock:
            self.__init__(self.max_latency_samples)
            print("📊 TextAnalysisMetrics reset")


# Global text analysis metrics instance
text_analysis_metrics = TextAnalysisMetrics()