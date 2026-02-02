# Text Analysis System Flow Documentation

Dokumentasi lengkap tentang bagaimana sistem Text Analysis bekerja ketika multiple jobs di-submit secara bersamaan.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detailed Flow](#detailed-flow)
4. [Timeline](#timeline)
5. [Data Flow](#data-flow)
6. [Error Handling](#error-handling)
7. [Metrics & Monitoring](#metrics--monitoring)

---

## Overview

Text Analysis System adalah microservices-based platform untuk character prescreening yang terdiri dari:

- **API Service**: Menerima request, validasi, dan submit job ke queue
- **Worker Service**: Memproses job dari queue dan memanggil AI model
- **Nexus (Open WebUI)**: AI platform dengan GPT-5 backend dan Google Search integration

### Key Components

| Component | Service | Purpose |
|-----------|---------|---------|
| API | `text-analysis-api-service` | REST API, auth, rate limiting |
| Worker | `text-analysis-worker-service` | Job processing, model calls |
| Queue | Pub/Sub `text-analysis-request` | Async message queue |
| Database | Firestore `text-analysis-firestore` | Job status & results |
| AI | Nexus (Open WebUI + GPT-5) | Text analysis with Google Search |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATION                                  │
│                                                                                  │
│   POST /api/analyze-text/negative-news                                          │
│   POST /api/analyze-text/pep-analysis                                           │
│   POST /api/analyze-text/law-involvement                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TEXT ANALYSIS API SERVICE                                │
│                    (Cloud Run: text-analysis-api-service)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│   │    Auth     │ → │ Rate Limit  │ → │  Sanitize   │ → │  Validate   │         │
│   │  (API Key)  │   │ (100/hour)  │   │   (XSS)     │   │  (Model)    │         │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                │                 │
│                                                                ▼                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         Create Job Record                                │   │
│   │   • Generate UUID                                                        │   │
│   │   • Save to Firestore (status: submitted)                               │   │
│   │   • Publish to Pub/Sub                                                  │   │
│   │   • Return job_id to client                                             │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │                                           │
                    ▼                                           ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│           FIRESTORE               │       │            PUB/SUB                │
│   (text-analysis-firestore)       │       │   Topic: text-analysis-request    │
├───────────────────────────────────┤       ├───────────────────────────────────┤
│                                   │       │                                   │
│   Collection: jobs                │       │   Subscription:                   │
│   ├── job-1 (submitted)           │       │   text-analysis-worker            │
│   ├── job-2 (submitted)           │       │                                   │
│   └── job-3 (submitted)           │       │   Messages in queue               │
│                                   │       │                                   │
└───────────────────────────────────┘       └───────────────────────────────────┘
                    ▲                                           │
                    │                                           ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │         TEXT ANALYSIS WORKER SERVICE          │
                    │               │      (Cloud Run: text-analysis-worker-service)│
                    │               ├───────────────────────────────────────────────┤
                    │               │                                               │
                    │               │   ┌─────────────────────────────────────────┐ │
                    │               │   │         Pub/Sub Subscriber              │ │
                    │               │   │   • Streaming pull                      │ │
                    │               │   │   • Flow control: max 16 messages       │ │
                    │               │   └─────────────────────────────────────────┘ │
                    │               │                       │                       │
                    │               │                       ▼                       │
                    │               │   ┌─────────────────────────────────────────┐ │
                    │               │   │      TextAnalysisProcessor              │ │
                    │               │   │   • Validate job data                   │ │
                    │               │   │   • Call model client                   │ │
                    │               │   │   • Format result                       │ │
                    │               │   └─────────────────────────────────────────┘ │
                    │               │                       │                       │
                    │               │                       ▼                       │
                    │               │   ┌─────────────────────────────────────────┐ │
                    │               │   │         TextModelClient                 │ │
                    │               │   │   • HTTP POST to Nexus                  │ │
                    │               │   │   • Retry logic (3 attempts)            │ │
                    │               │   │   • Timeout: 300s                       │ │
                    │               │   └─────────────────────────────────────────┘ │
                    │               │                       │                       │
                    │               └───────────────────────│───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │              NEXUS (Open WebUI)               │
                    │               │         Backend: GPT-5 + Google Search        │
                    │               ├───────────────────────────────────────────────┤
                    │               │                                               │
                    │               │   Models:                                     │
                    │               │   ├── negative-news                           │
                    │               │   ├── politically-exposed-person-v2           │
                    │               │   ├── law-involvement                         │
                    │               │   ├── negative-news-corporate                 │
                    │               │   └── law-involvement-corporate               │
                    │               │                                               │
                    │               │   Tools: web_search_with_google               │
                    │               │                                               │
                    │               └───────────────────────────────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │            GOOGLE SEARCH API                  │
                    │               └───────────────────────────────────────────────┘
                    │                                       │
                    └───────────────────────────────────────┘
                              Response saved to Firestore
```

---

## Detailed Flow

### Phase 1: Client Submit Request

Client mengirim HTTP request ke API Service:

```http
POST /api/analyze-text/negative-news
Content-Type: application/json

{
  "name": "Ratna Juwita",
  "entity_type": "person"
}
```

### Phase 2: API Service Processing


#### 2.1 Authentication & Rate Limiting

```python
# Check API key (if REQUIRE_AUTH=true)
if AuthConfig.REQUIRE_AUTH:
    if credentials.credentials not in AuthConfig.VALID_API_KEYS:
        raise HTTPException(401, "Invalid API key")

# Rate limiting - sliding window algorithm
# Default: 100 requests per 3600 seconds (1 hour)
allowed, rate_info = rate_limiter.is_allowed(client_id)
if not allowed:
    raise HTTPException(429, "Rate limit exceeded")
```

#### 2.2 Input Validation & Sanitization

```python
class TextAnalysisRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    entity_type: EntityType  # "person" or "corporate"
    
    @validator('name')
    def validate_name(cls, v):
        # Sanitize untuk prevent XSS, SQL injection
        return InputSanitizer.sanitize_name(v, "name")
```

**InputSanitizer melakukan:**
- Strip whitespace
- Remove HTML tags
- Escape special characters
- Block SQL injection patterns
- Block XSS patterns

#### 2.3 Model Configuration

```python
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
        "entity_types": ["corporate"]
    },
    "corporate-law-involvement": {
        "model": "law-involvement-corporate",
        "entity_types": ["corporate"]
    }
}
```

#### 2.4 Create Job Record (Firestore)

```python
job_data = {
    "job_id": "abc-123-xxx",           # UUID unik
    "job_type": "text_analysis",
    "status": "submitted",              # Initial status
    "analysis_type": "negative-news",
    "entity_type": "person",
    "name": "Ratna Juwita",
    "model_name": "negative-news",
    "created_at": datetime.now(UTC),
    "result": None,
    "error": None
}

firestore_client.collection("jobs").document(job_id).set(job_data)
```

#### 2.5 Publish to Pub/Sub

```python
message_data = {
    "job_id": job_id,
    "job_type": "text_analysis",
    "analysis_type": analysis_type,
    "entity_type": entity_type,
    "name": name,
    "model_name": model_config["model"],
    "timestamp": datetime.now(UTC).isoformat()
}

topic_path = publisher.topic_path(PROJECT_ID, "text-analysis-request")
publisher.publish(topic_path, json.dumps(message_data).encode('utf-8'))
```

#### 2.6 Return Response

```json
{
  "success": true,
  "job_id": "abc-123-xxx",
  "status": "submitted",
  "analysis_type": "negative-news",
  "entity_type": "person",
  "name": "Ratna Juwita",
  "model_name": "negative-news",
  "submitted_at": "2026-01-27T10:00:00Z",
  "processing_time": 0.05,
  "message": "Text analysis job submitted successfully"
}
```

### Phase 3: Pub/Sub Message Delivery

```
Topic: text-analysis-request
Subscription: text-analysis-worker (Pull mode)

Queue State:
├── Message 1: {job_id: "abc-123", name: "Ratna Juwita", ...}
├── Message 2: {job_id: "def-456", name: "Deisti Astriani Tagor", ...}
└── Message 3: {job_id: "ghi-789", name: "Astri Arini", ...}
```

Worker menggunakan streaming pull dengan flow control:

```python
flow_control = pubsub_v1.types.FlowControl(
    max_messages=16,              # Max 16 messages at a time
    max_bytes=100 * 1024 * 1024   # 100MB max
)

streaming_pull_future = subscriber.subscribe(
    subscription_path,
    callback=process_message,
    flow_control=flow_control
)
```

### Phase 4: Worker Processing

#### 4.1 Message Reception

```python
def process_message(self, message):
    message_data = json.loads(message.data.decode("utf-8"))
    job_id = message_data.get("job_id")
    job_type = message_data.get("job_type")
    
    # Track active job
    self.active_jobs[job_id] = {
        "start_time": datetime.now(UTC),
        "job_type": job_type
    }
```

#### 4.2 Update Status to Processing

```python
def update_job_status(self, job_id, status, result=None, error=None):
    doc_ref = firestore_client.collection("jobs").document(job_id)
    
    update_data = {
        "status": status,
        "updated_at": datetime.now(UTC),
    }
    
    if status == "completed":
        update_data["completed_at"] = datetime.now(UTC)
        update_data["result"] = result
    elif status == "failed":
        update_data["completed_at"] = datetime.now(UTC)
        update_data["error"] = error
    
    doc_ref.update(update_data)
```

#### 4.3 Process Text Analysis

```python
def process_text_analysis(self, job_data):
    job_id = job_data.get("job_id")
    analysis_type = job_data.get("analysis_type")
    name = job_data.get("name")
    
    # Get model config
    model_config = self.text_model_config[analysis_type]
    model_name = model_config["model"]
    
    # Call AI model
    analysis_result = self.call_text_analysis_model(
        model_name=model_name,
        name=name,
        analysis_type=analysis_type,
        entity_type=entity_type
    )
    
    # Format result
    return self.format_analysis_result(analysis_result, ...)
```

### Phase 5: Model Client HTTP Request

#### 5.1 Request to Nexus

```python
def _make_request(self, model_name, prompt, attempt, **kwargs):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "tool_ids": ["web_search_with_google"]
    }
    # Note: NO temperature, NO max_tokens - vanilla call
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
    }
    
    response = requests.post(
        self.chat_endpoint,
        headers=headers,
        json=payload,
        timeout=300  # 5 minutes
    )
```

**Actual HTTP Request:**

```http
POST https://nexus-bnimove-xxx.run.app/api/chat/completions
Authorization: Bearer sk-xxx
Content-Type: application/json

{
  "model": "negative-news",
  "messages": [
    {"role": "user", "content": "Ratna Juwita"}
  ],
  "stream": false,
  "tool_ids": ["web_search_with_google"]
}
```

#### 5.2 Retry Logic

```python
def call_model(self, model_name, prompt, **kwargs):
    for attempt in range(self.max_retries + 1):  # 4 attempts total
        try:
            return self._make_request(model_name, prompt, attempt, **kwargs)
        except Exception as e:
            if not self._is_retryable(e) or attempt >= self.max_retries:
                break
            
            # Exponential backoff with jitter
            delay = self.retry_delay * (2 ** attempt) * random.uniform(0.5, 1.5)
            delay = min(delay, self.max_retry_delay)  # Cap at 60s
            time.sleep(delay)
    
    raise Exception(f"Failed after {self.max_retries + 1} attempts")
```

**Retryable errors:** timeout, connection, rate limit, 429, 500, 502, 503, 504

**Non-retryable errors:** 400, 401, 403, 404, reasoning token limit

### Phase 6: Nexus (Open WebUI) Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEXUS (Open WebUI)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Receive request                                              │
│     └── Model: negative-news                                     │
│     └── Input: "Ratna Juwita"                                    │
│                                                                  │
│  2. Load model configuration                                     │
│     └── System prompt: (built-in instructions)                   │
│     └── Max tokens: 16000 (configured in Open WebUI)             │
│                                                                  │
│  3. GPT-5 Processing                                             │
│     ├── Parse input name                                         │
│     ├── Generate search queries:                                 │
│     │   └── "Ratna Juwita korupsi"                               │
│     │   └── "Ratna Juwita kasus hukum"                           │
│     │   └── "Ratna Juwita berita negatif"                        │
│     │                                                            │
│     ├── Call Google Search API (via tool)                        │
│     │   └── Search 1: 10 results                                 │
│     │   └── Search 2: 10 results                                 │
│     │   └── Search 3: 10 results                                 │
│     │                                                            │
│     ├── Analyze search results                                   │
│     │   └── Read snippets                                        │
│     │   └── Identify negative news                               │
│     │   └── Extract key information                              │
│     │                                                            │
│     ├── Reasoning (internal)                                     │
│     │   └── Uses reasoning tokens                                │
│     │   └── Can use up to 2000+ reasoning tokens                 │
│     │                                                            │
│     └── Generate response                                        │
│         └── Structured JSON with findings                        │
│                                                                  │
│  4. Return response to Worker (~2 minutes)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 7: Response Processing

#### 7.1 Parse Response

```python
response_data = response.json()
message = response_data["choices"][0]["message"]
content = message["content"]

# Validate content
if not content or content.strip() == "":
    usage = response_data.get("usage", {})
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if reasoning_tokens >= 2000:
        raise Exception("Model hit reasoning token limit - empty response")
    raise Exception("Model returned empty content")
```

#### 7.2 Format Result

```python
def format_analysis_result(self, analysis_result, ...):
    content = analysis_result.get("content", "")
    parsed_result = self.extract_json_from_content(content)
    
    return {
        "analysis_type": analysis_type,
        "entity_type": entity_type,
        "entity_name": name,
        "model_used": model_name,
        "findings": {
            "status": self.extract_status(parsed_result),
            "summary": self.extract_summary(parsed_result),
            "details": self.extract_details(parsed_result),
            "sources": self.extract_sources(parsed_result),
            "last_updated": datetime.now(UTC).isoformat()
        },
        "metadata": {
            "processing_time": analysis_result.get("response_time", 0),
            "model_version": model_name,
            "usage": analysis_result.get("usage", {})
        },
        "raw_response": content
    }
```

**Status extraction logic:**

| Model Response | Extracted Status | Meaning |
|----------------|------------------|---------|
| "positive", "yes", "found" | `positive` | Negative news FOUND |
| "negative", "no", "clean" | `negative` | Clean, no issues |
| "neutral", "unclear" | `neutral` | Inconclusive |
| (other) | `unknown` | Cannot determine |

### Phase 8: Save Result & Acknowledge

```python
# Update Firestore
if result.get("success"):
    self.update_job_status(job_id, "completed", result=result.get("result"))
else:
    self.update_job_status(job_id, "failed", error=result.get("error"))

# Acknowledge Pub/Sub message
message.ack()
```

---

## Timeline

### Example: 3 Jobs Submitted Simultaneously


```
T+0.000s  │ Client submit Job 1 (Ratna Juwita - negative-news)
T+0.001s  │ Client submit Job 2 (Deisti Astriani Tagor - negative-news)
T+0.002s  │ Client submit Job 3 (Astri Arini - pep-analysis)
          │
T+0.050s  │ API: Job 1 validated, Firestore created, Pub/Sub published
T+0.051s  │ API: Job 2 validated, Firestore created, Pub/Sub published
T+0.052s  │ API: Job 3 validated, Firestore created, Pub/Sub published
          │
T+0.100s  │ API returns response for all 3 jobs (status: submitted)
          │
          │ ═══════════════════════════════════════════════════════
          │
T+0.200s  │ Worker receives Message 1 (Job 1)
T+0.201s  │ Worker receives Message 2 (Job 2)  ← Queued
T+0.202s  │ Worker receives Message 3 (Job 3)  ← Queued
          │
T+0.300s  │ Worker: Job 1 status → "processing"
T+0.400s  │ Worker: Calling Nexus for Job 1
          │
          │ ─────────── Nexus Processing Job 1 (~2 minutes) ───────────
          │ │ GPT-5 parsing "Ratna Juwita"
          │ │ Google Search: "Ratna Juwita korupsi"
          │ │ Google Search: "Ratna Juwita kasus hukum"
          │ │ Analyzing results...
          │ │ Generating response...
          │
T+120.0s  │ Worker: Job 1 response received
T+120.1s  │ Worker: Job 1 status → "completed"
T+120.2s  │ Worker: Job 1 message acknowledged
          │
          │ ═══════════════════════════════════════════════════════
          │
T+120.3s  │ Worker: Job 2 starts processing
T+120.4s  │ Worker: Job 2 status → "processing"
T+120.5s  │ Worker: Calling Nexus for Job 2
          │
          │ ─────────── Nexus Processing Job 2 (~2 minutes) ───────────
          │ │ GPT-5 parsing "Deisti Astriani Tagor"
          │ │ Google Search: "Deisti Astriani Tagor KPK"
          │ │ Found: KPK e-KTP case involvement
          │ │ Analyzing results...
          │
T+240.0s  │ Worker: Job 2 response received
T+240.1s  │ Worker: Job 2 status → "completed"
T+240.2s  │ Worker: Job 2 message acknowledged
          │
          │ ═══════════════════════════════════════════════════════
          │
T+240.3s  │ Worker: Job 3 starts processing
T+240.4s  │ Worker: Job 3 status → "processing"
T+240.5s  │ Worker: Calling Nexus for Job 3
          │
          │ ─────────── Nexus Processing Job 3 (~2 minutes) ───────────
          │ │ GPT-5 parsing "Astri Arini"
          │ │ Google Search: "Astri Arini PEP"
          │ │ Google Search: "Astri Arini pejabat politik"
          │ │ Result: No PEP records found
          │
T+360.0s  │ Worker: Job 3 response received
T+360.1s  │ Worker: Job 3 status → "completed"
T+360.2s  │ Worker: Job 3 message acknowledged
          │
          │ ═══════════════════════════════════════════════════════
          │
T+360.3s  │ ALL 3 JOBS COMPLETED!
```

**Summary:**
- Total time: ~6 minutes (360 seconds)
- Average per job: ~2 minutes
- Processing: Sequential (not parallel)

### Why Sequential Processing?

| Factor | Description |
|--------|-------------|
| Single Worker Instance | Cloud Run typically runs 1 instance |
| Long Model Response | GPT-5 + Google Search = ~2 min per request |
| Nexus Bottleneck | Open WebUI may process requests sequentially |
| Resource Constraints | GPT-5 API and Google Search rate limits |

**Breakdown waktu per job:**

| Step | Duration |
|------|----------|
| Network latency Worker → Nexus | ~0.1s |
| Nexus parsing & routing | ~0.5s |
| GPT-5 understanding query | ~2s |
| Google Search calls (3-5x) | ~10s |
| GPT-5 reading search results | ~30s |
| GPT-5 reasoning | ~60s |
| GPT-5 generating response | ~20s |
| Network latency Nexus → Worker | ~0.1s |
| **Total** | **~120s (2 minutes)** |

---

## Data Flow

### Complete Data Flow for Single Job

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           JOB: Ratna Juwita (negative-news)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT:                                                                          │
│  {                                                                               │
│    "name": "Ratna Juwita",                                                       │
│    "entity_type": "person",                                                      │
│    "analysis_type": "negative-news"                                              │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  FIRESTORE RECORD (Initial):                                                     │
│  {                                                                               │
│    "job_id": "abc-123-xxx",                                                      │
│    "job_type": "text_analysis",                                                  │
│    "status": "submitted",                                                        │
│    "analysis_type": "negative-news",                                             │
│    "entity_type": "person",                                                      │
│    "name": "Ratna Juwita",                                                       │
│    "model_name": "negative-news",                                                │
│    "created_at": "2026-01-27T10:00:00.050Z",                                     │
│    "result": null,                                                               │
│    "error": null                                                                 │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  PUB/SUB MESSAGE:                                                                │
│  {                                                                               │
│    "job_id": "abc-123-xxx",                                                      │
│    "job_type": "text_analysis",                                                  │
│    "analysis_type": "negative-news",                                             │
│    "entity_type": "person",                                                      │
│    "name": "Ratna Juwita",                                                       │
│    "model_name": "negative-news",                                                │
│    "timestamp": "2026-01-27T10:00:00.051Z"                                       │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  HTTP REQUEST TO NEXUS:                                                          │
│  POST /api/chat/completions                                                      │
│  {                                                                               │
│    "model": "negative-news",                                                     │
│    "messages": [{"role": "user", "content": "Ratna Juwita"}],                    │
│    "stream": false,                                                              │
│    "tool_ids": ["web_search_with_google"]                                        │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  NEXUS RESPONSE:                                                                 │
│  {                                                                               │
│    "choices": [{                                                                 │
│      "message": {                                                                │
│        "content": "{\"status\": \"positive\", \"summary\": \"Ditemukan          │
│                    berita negatif terkait dugaan korupsi...\", ...}"            │
│      }                                                                           │
│    }],                                                                           │
│    "usage": {                                                                    │
│      "prompt_tokens": 150,                                                       │
│      "completion_tokens": 2500,                                                  │
│      "completion_tokens_details": {"reasoning_tokens": 1800}                     │
│    },                                                                            │
│    "sources": [                                                                  │
│      {"url": "https://news.detik.com/...", "title": "..."},                      │
│      {"url": "https://kompas.com/...", "title": "..."}                           │
│    ]                                                                             │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  FORMATTED RESULT:                                                               │
│  {                                                                               │
│    "analysis_type": "negative-news",                                             │
│    "entity_type": "person",                                                      │
│    "entity_name": "Ratna Juwita",                                                │
│    "model_used": "negative-news",                                                │
│    "findings": {                                                                 │
│      "status": "positive",           ← FOUND negative news                       │
│      "summary": "Ditemukan berita negatif terkait dugaan korupsi...",            │
│      "details": [                                                                │
│        {                                                                         │
│          "type": "corruption",                                                   │
│          "description": "Terlibat dalam kasus korupsi...",                       │
│          "source": "detik.com"                                                   │
│        }                                                                         │
│      ],                                                                          │
│      "sources": ["https://news.detik.com/...", "https://kompas.com/..."],        │
│      "last_updated": "2026-01-27T10:02:15Z"                                      │
│    },                                                                            │
│    "metadata": {                                                                 │
│      "processing_time": 120.5,                                                   │
│      "model_version": "negative-news",                                           │
│      "usage": {"prompt_tokens": 150, "completion_tokens": 2500}                  │
│    }                                                                             │
│  }                                                                               │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  FIRESTORE RECORD (Final):                                                       │
│  {                                                                               │
│    "job_id": "abc-123-xxx",                                                      │
│    "status": "completed",                                                        │
│    "completed_at": "2026-01-27T10:02:15Z",                                       │
│    "result": { ... formatted result above ... },                                 │
│    "error": null                                                                 │
│  }                                                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### Scenario 1: Model Timeout

```
Worker → Nexus: POST (timeout 300s)
         ↓
         ... 300 seconds pass ...
         ↓
Worker: requests.exceptions.Timeout
         ↓
Worker: _is_retryable("timeout") → True
         ↓
Worker: Retry attempt 1 (delay: 1s)
Worker: Retry attempt 2 (delay: 2s)
Worker: Retry attempt 3 (delay: 4s)
         ↓
Worker: All retries failed
         ↓
Firestore: status = "failed", error = "Model call failed after 4 attempts"
```

### Scenario 2: Empty Response (Reasoning Token Limit)

```
Nexus Response:
{
  "choices": [{"message": {"content": ""}}],  ← Empty!
  "usage": {
    "completion_tokens_details": {
      "reasoning_tokens": 2000  ← Hit limit!
    }
  }
}
         ↓
Worker: content.strip() == ""
Worker: reasoning_tokens >= 2000
         ↓
Worker: raise Exception("Model hit reasoning token limit")
         ↓
Worker: _is_retryable("reasoning token limit") → False  ← No retry!
         ↓
Firestore: status = "failed", error = "Model hit reasoning token limit"
```

### Scenario 3: Rate Limit (429)

```
Nexus Response: HTTP 429 Too Many Requests
         ↓
Worker: _is_retryable("429") → True
         ↓
Worker: Retry with exponential backoff
        Attempt 1: wait 1s
        Attempt 2: wait 2s
        Attempt 3: wait 4s
         ↓
If still 429 after all retries:
Firestore: status = "failed", error = "Rate limit exceeded"
```

### Scenario 4: Invalid Job Type

```
Pub/Sub Message: {"job_type": "document_processing", ...}  ← Wrong type!
         ↓
Worker: if job_type != "text_analysis":
         ↓
Firestore: status = "failed", error = "Invalid job type"
Worker: message.ack()  ← Still ack to prevent redelivery
```

### Error Response Codes

| HTTP Code | Error Type | Retryable |
|-----------|------------|-----------|
| 400 | Bad Request | No |
| 401 | Unauthorized | No |
| 403 | Forbidden | No |
| 404 | Not Found | No |
| 429 | Rate Limit | Yes |
| 500 | Server Error | Yes |
| 502 | Bad Gateway | Yes |
| 503 | Service Unavailable | Yes |
| 504 | Gateway Timeout | Yes |
| Timeout | Connection Timeout | Yes |

---

## Metrics & Monitoring

### Worker Metrics

```json
{
  "total_jobs_processed": 150,
  "total_jobs_failed": 5,
  "success_rate": 96.7,
  
  "by_analysis_type": {
    "negative-news": {
      "count": 80,
      "avg_time": 125.5,
      "success_rate": 97.5
    },
    "pep-analysis": {
      "count": 50,
      "avg_time": 140.2,
      "success_rate": 96.0
    },
    "law-involvement": {
      "count": 20,
      "avg_time": 118.3,
      "success_rate": 95.0
    }
  },
  
  "error_breakdown": {
    "TIMEOUT": 2,
    "PROCESSING_ERROR": 2,
    "EXCEPTION": 1
  }
}
```

### API Metrics

```json
{
  "overview": {
    "total_requests": 500,
    "success_rate": 98.5,
    "avg_latency_ms": 52,
    "requests_per_minute": 2.5,
    "uptime_seconds": 86400
  },
  
  "by_analysis_type": {
    "negative-news": {"requests": 250, "success_rate": 99.0},
    "pep-analysis": {"requests": 150, "success_rate": 98.0},
    "law-involvement": {"requests": 100, "success_rate": 97.5}
  },
  
  "model_availability": {
    "negative-news": {"available": true},
    "politically-exposed-person-v2": {"available": true}
  }
}
```

### Health Check Endpoints

**API Service:** `GET /health`

```json
{
  "status": "healthy",
  "checks": {
    "gcs": "✅ accessible",
    "firestore": "✅ accessible",
    "pubsub": "✅ accessible",
    "text_models": "✅ 5/5 models healthy"
  }
}
```

**Worker Service:** `GET /health`

```json
{
  "status": "healthy",
  "worker": {
    "status": "running",
    "uptime_seconds": 3600,
    "processed_jobs": 50,
    "failed_jobs": 2,
    "active_jobs": 1
  }
}
```

---

## Client Polling

Client dapat mengecek status job via:

```http
GET /api/status/{job_id}
```

**Response saat processing:**

```json
{
  "job_id": "abc-123-xxx",
  "status": "processing",
  "analysis_type": "negative-news",
  "name": "Ratna Juwita",
  "created_at": "2026-01-27T10:00:00Z",
  "updated_at": "2026-01-27T10:00:05Z"
}
```

**Response saat completed:**

```json
{
  "job_id": "abc-123-xxx",
  "status": "completed",
  "analysis_type": "negative-news",
  "name": "Ratna Juwita",
  "created_at": "2026-01-27T10:00:00Z",
  "completed_at": "2026-01-27T10:02:15Z",
  "result": {
    "findings": {
      "status": "positive",
      "summary": "Ditemukan berita negatif...",
      "details": [...],
      "sources": [...]
    }
  }
}
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| Architecture | Microservices (API + Worker) |
| Queue | Google Pub/Sub |
| Database | Google Firestore |
| AI Backend | Nexus (Open WebUI) + GPT-5 |
| Search | Google Search API |
| Processing | Sequential (~2 min/job) |
| Retry | 3 attempts with exponential backoff |
| Timeout | 300 seconds (5 minutes) |

**Key Insight:** Bottleneck ada di Nexus/GPT-5 + Google Search, bukan di Worker atau Pub/Sub. Untuk parallel processing, perlu multiple worker instances atau model yang lebih cepat.
