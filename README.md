# 📚 Deployment Architecture - Text Doc Services

Dokumentasi lengkap tentang konsep dan arsitektur deployment sistem Text Document Processing & Analysis.

---

## 🎯 Overview

Sistem ini menggunakan **microservices architecture** dengan **CI/CD automation** untuk deployment ke Google Cloud Platform.

### Komponen Utama:
1. **API Service** - REST API untuk menerima request
2. **Worker Service** - Background processor untuk document & text analysis
3. **Cloud Build** - CI/CD pipeline untuk auto-deployment
4. **GitHub** - Source code repository & version control

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                         GITHUB                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ text-doc-api-service     │  │ text-doc-worker-service  │   │
│  │ - app.py                 │  │ - worker.py              │   │
│  │ - cloudbuild.yaml        │  │ - cloudbuild.yaml        │   │
│  │ - Dockerfile             │  │ - Dockerfile             │   │
│  │ - requirements.txt       │  │ - requirements.txt       │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │ git push                   │ git push
              │ (trigger)                  │ (trigger)
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD BUILD TRIGGERS                         │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ API Service Trigger      │  │ Worker Service Trigger   │   │
│  │ - Branch: main           │  │ - Branch: main           │   │
│  │ - Config: cloudbuild.yaml│  │ - Config: cloudbuild.yaml│   │
│  │ - Region: asia-southeast2│  │ - Region: asia-southeast2│   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │ execute build              │ execute build
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD BUILD                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Build Steps:                                             │  │
│  │ 1. Build Docker image                                    │  │
│  │ 2. Push to Artifact Registry (asia-southeast2)          │  │
│  │ 3. Deploy to Cloud Run                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────┬────────────────────────────┬────────────────────┘
              │ deploy                     │ deploy
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ARTIFACT REGISTRY                          │
│  asia-southeast2-docker.pkg.dev/PROJECT_ID/cloud-run-source-... │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ API Service Images       │  │ Worker Service Images    │   │
│  │ - :latest                │  │ - :latest                │   │
│  │ - :commit-sha            │  │ - :commit-sha            │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │ pull & run                 │ pull & run
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD RUN                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ text-doc-api-service-v2  │  │ text-doc-worker-service-v2│  │
│  │ - 2Gi RAM, 1 CPU         │  │ - 4Gi RAM, 2 CPU         │   │
│  │ - 0-10 instances         │  │ - 1-10 instances         │   │
│  │ - Port: 8080             │  │ - Port: 8080             │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │                            │
              │ ┌──────────────────────────┴──────────┐
              │ │     GOOGLE CLOUD SERVICES           │
              ├─┤ - Firestore (database)              │
              ├─┤ - Cloud Storage (file storage)      │
              ├─┤ - Pub/Sub (messaging)                │
              └─┤ - Secret Manager (credentials)       │
                └─────────────────────────────────────┘
```

---

## 🔄 Deployment Flow

### 0. Initial Deployment (Deployment Pertama Kali)

**PENTING**: Sebelum Cloud Build Trigger bisa bekerja, service harus dibuat dulu secara manual.

#### Kenapa Perlu Manual Deploy Pertama?

Cloud Build Trigger **TIDAK BISA** membuat service baru karena:
- Trigger hanya execute `cloudbuild.yaml`
- Di `cloudbuild.yaml` ada command: `gcloud run deploy SERVICE_NAME`
- Command ini **update** service yang sudah ada, bukan create new service
- Jika service belum ada, deployment akan **GAGAL**

#### Cara 1: Manual Deployment (Recommended untuk Pertama Kali)

```bash
# 1. Masuk ke directory service
cd text-doc-api-service

# 2. Deploy dengan gcloud CLI
gcloud run deploy text-doc-api-service-v2 \
  --source . \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai \
  --allow-unauthenticated

# 3. Tunggu proses selesai (5-10 menit)
# Output akan menampilkan URL service
```

**Yang Terjadi di Balik Layar**:

```
Step 1: gcloud CLI membaca source code lokal
        ↓
Step 2: Otomatis trigger Cloud Build untuk build Docker image
        ↓
Step 3: Build Docker image dari Dockerfile
        ↓
Step 4: Push image ke Artifact Registry
        - Location: asia-southeast2-docker.pkg.dev
        - Repository: cloud-run-source-deploy
        - Tag: latest
        ↓
Step 5: BUAT Cloud Run service baru (PENTING!)
        - Service name: text-doc-api-service-v2
        - Region: asia-southeast2
        - Generate URL otomatis
        ↓
Step 6: Deploy revision pertama
        - Pull image dari Artifact Registry
        - Start container
        - Allocate resources (RAM, CPU)
        ↓
Step 7: Service LIVE dengan URL publik
        URL: https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app
```

#### Cara 2: Deploy via Console (Alternative)

1. Buka Cloud Run Console: https://console.cloud.google.com/run
2. Click "CREATE SERVICE"
3. Pilih "Continuously deploy from a repository (source or function)"
4. Connect GitHub repository
5. Configure build settings:
   - Branch: main
   - Build type: Dockerfile
   - Dockerfile path: /Dockerfile
6. Configure service settings:
   - Service name: text-doc-api-service-v2
   - Region: asia-southeast2
   - CPU allocation: CPU is always allocated
   - Memory: 2Gi
   - CPU: 1
7. Click "CREATE"

#### Komponen yang Dibuat Otomatis

Saat deployment pertama kali, Google Cloud **otomatis membuat**:

1. **Cloud Run Service**
   - Nama: text-doc-api-service-v2
   - Region: asia-southeast2
   - URL: https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app
   - Status: LIVE

2. **Artifact Registry Repository** (jika belum ada)
   - Nama: cloud-run-source-deploy
   - Location: asia-southeast2-docker.pkg.dev
   - Format: Docker

3. **Service Account** (jika belum ada)
   - Untuk menjalankan service
   - Permissions: Firestore, Storage, Pub/Sub
   - Default: PROJECT_NUMBER-compute@developer.gserviceaccount.com

4. **Load Balancer & SSL Certificate**
   - Untuk handle HTTPS traffic
   - SSL certificate otomatis di-provision
   - Managed oleh Google Cloud

5. **First Revision**
   - Revision name: SERVICE_NAME-00001-xxx
   - Image: IMAGE_URL:latest
   - Traffic: 100%

#### Setelah Initial Deployment

Setelah service berhasil dibuat, baru setup Cloud Build Trigger:

```bash
# 1. Buka Cloud Build Triggers Console
# https://console.cloud.google.com/cloud-build/triggers

# 2. Create Trigger dengan konfigurasi:
# - Name: text-doc-api-service-v2-auto-deploy
# - Repository: cotr46/text-doc-api-service
# - Branch: ^main$
# - Build configuration: cloudbuild.yaml
# - Region: asia-southeast2

# 3. Sekarang setiap push ke main akan auto-deploy
```

#### Urutan Lengkap Setup Baru

```
1. INITIAL DEPLOYMENT (Manual)
   ├─ gcloud run deploy --source .
   ├─ Service dibuat
   ├─ URL di-generate
   └─ Service LIVE
   
2. SETUP CLOUD BUILD TRIGGER
   ├─ Create trigger via Console
   ├─ Connect ke GitHub repo
   ├─ Configure cloudbuild.yaml
   └─ Trigger ACTIVE
   
3. DEPLOYMENT SELANJUTNYA (Otomatis)
   ├─ Push code ke GitHub
   ├─ Trigger activated
   ├─ Build & deploy otomatis
   └─ Service updated
```

#### Perbedaan Manual vs Auto Deploy

| Aspek | Manual Deploy (Pertama) | Auto Deploy (Selanjutnya) |
|-------|-------------------------|---------------------------|
| **Trigger** | Command `gcloud run deploy` | Git push ke GitHub |
| **Service** | **BUAT service baru** | Update service existing |
| **Build** | Otomatis via gcloud | Via Cloud Build Trigger |
| **Image Tag** | latest | commit SHA + latest |
| **Revision** | Revision pertama | Revision baru |
| **URL** | Generate URL baru | Pakai URL existing |
| **Durasi** | 5-10 menit | 5-10 menit |

#### Verifikasi Initial Deployment

```bash
# 1. Check service status
gcloud run services describe text-doc-api-service-v2 \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai

# 2. Test health endpoint
curl https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app/health

# 3. View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=text-doc-api-service-v2" \
  --limit=20 \
  --project=bni-prod-dma-bnimove-ai

# 4. List revisions
gcloud run revisions list \
  --service=text-doc-api-service-v2 \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai
```

---

### 1. Developer Workflow (Setelah Initial Setup)

```bash
# Developer membuat perubahan di local
cd text-doc-api-service
vim app.py  # Edit code

# Commit changes
git add .
git commit -m "Add new feature"

# Push ke GitHub
git push origin main
```

### 2. Automatic Trigger

Saat code di-push ke GitHub branch `main`:

1. **GitHub webhook** mengirim notifikasi ke Cloud Build
2. **Cloud Build Trigger** terdeteksi ada push baru
3. **Build process** dimulai otomatis

### 3. Build Process

Cloud Build menjalankan steps di `cloudbuild.yaml`:

```yaml
steps:
  # Step 1: Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'IMAGE_URL', '.']
    
  # Step 2: Push image dengan commit SHA
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'IMAGE_URL:COMMIT_SHA']
    
  # Step 3: Push image dengan tag latest
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'IMAGE_URL:latest']
    
  # Step 4: Deploy ke Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: ['gcloud', 'run', 'deploy', 'SERVICE_NAME', 
           '--image=IMAGE_URL:COMMIT_SHA',
           '--region=asia-southeast2']
```

**Durasi**: 5-10 menit

### 4. Deployment ke Cloud Run

Setelah build selesai:

1. **Cloud Run** pull Docker image dari Artifact Registry
2. **Create new revision** dengan image baru
3. **Route traffic** ke revision baru (100%)
4. **Old revision** tetap ada (untuk rollback jika perlu)
5. **Service ready** dengan code terbaru

---

## 📦 Komponen Detail

### A. GitHub Repositories

**Struktur**:
```
text-doc-api-service/
├── app.py                    # Main API application
├── auth.py                   # Authentication & authorization
├── security.py               # Security utilities
├── text_analysis_metrics.py  # Metrics tracking
├── cloudbuild.yaml           # CI/CD configuration
├── Dockerfile                # Container image definition
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation

text-doc-worker-service/
├── worker.py                 # Main worker application
├── pdf_processor.py          # PDF processing logic
├── text_analysis_processor.py # Text analysis logic
├── text_model_client.py      # AI model client
├── cloudbuild.yaml           # CI/CD configuration
├── Dockerfile                # Container image definition
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

**URLs**:
- API: https://github.com/cotr46/text-doc-api-service
- Worker: https://github.com/cotr46/text-doc-worker-service

---

### B. Cloud Build Triggers

**Configuration**:

| Property | API Service | Worker Service |
|----------|-------------|----------------|
| Name | text-doc-api-service-v2-auto-deploy | text-doc-worker-service-v2-auto-deploy |
| Repository | cotr46/text-doc-api-service | cotr46/text-doc-worker-service |
| Branch | ^main$ | ^main$ |
| Build Config | cloudbuild.yaml | cloudbuild.yaml |
| Region | asia-southeast2 | asia-southeast2 |

**Trigger Event**: Push to branch `main`

**View Triggers**:
```bash
gcloud builds triggers list --project=bni-prod-dma-bnimove-ai
```

---

### C. Cloud Build

**Build Configuration** (`cloudbuild.yaml`):

```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: [
      'build',
      '-t', 'asia-southeast2-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/SERVICE_NAME:$COMMIT_SHA',
      '-t', 'asia-southeast2-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/SERVICE_NAME:latest',
      '.'
    ]
    id: 'build-service'

  # Push image with commit SHA
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'IMAGE_URL:$COMMIT_SHA']
    id: 'push-service'
    waitFor: ['build-service']

  # Push latest tag
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'IMAGE_URL:latest']
    id: 'push-service-latest'
    waitFor: ['build-service']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: [
      'gcloud', 'run', 'deploy', 'SERVICE_NAME',
      '--image=IMAGE_URL:$COMMIT_SHA',
      '--region=asia-southeast2',
      '--platform=managed'
    ]
    id: 'deploy-service'
    waitFor: ['push-service']

images:
  - 'IMAGE_URL:$COMMIT_SHA'
  - 'IMAGE_URL:latest'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'
```

**Variables**:
- `$PROJECT_ID`: bni-prod-dma-bnimove-ai
- `$COMMIT_SHA`: Git commit hash (unique per commit)

**View Builds**:
```bash
gcloud builds list --project=bni-prod-dma-bnimove-ai --limit=5
```

---

### D. Artifact Registry

**Registry URL**:
```
asia-southeast2-docker.pkg.dev/bni-prod-dma-bnimove-ai/cloud-run-source-deploy
```

**Images**:
- `text-doc-api-service-v2:latest`
- `text-doc-api-service-v2:COMMIT_SHA`
- `text-doc-worker-service-v2:latest`
- `text-doc-worker-service-v2:COMMIT_SHA`

**Why asia-southeast2?**
- Organization policy requirement
- Compliance dengan data residency
- Lower latency untuk region Indonesia

**View Images**:
```bash
gcloud artifacts docker images list \
  asia-southeast2-docker.pkg.dev/bni-prod-dma-bnimove-ai/cloud-run-source-deploy
```

---

### E. Cloud Run Services

**API Service v2**:
```yaml
Name: text-doc-api-service-v2
URL: https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app
Region: asia-southeast2
Resources:
  Memory: 2Gi
  CPU: 1
  Min instances: 0
  Max instances: 10
Port: 8080
```

**Worker Service v2**:
```yaml
Name: text-doc-worker-service-v2
URL: https://text-doc-worker-service-v2-lh5pr6ewdq-et.a.run.app
Region: asia-southeast2
Resources:
  Memory: 4Gi
  CPU: 2
  Min instances: 1
  Max instances: 10
Port: 8080
```

**View Services**:
```bash
gcloud run services list --region=asia-southeast2 --project=bni-prod-dma-bnimove-ai
```

---

## 🔐 Security & Compliance

### 1. Organization Policy

**Requirement**: Docker images harus di region `asia-southeast2`

**Implementation**:
- ✅ Artifact Registry: `asia-southeast2-docker.pkg.dev`
- ✅ Cloud Run: `asia-southeast2`
- ❌ Tidak boleh: `gcr.io` (multi-region, includes "us")

### 2. Authentication

**API Service**:
- Rate limiting
- Input sanitization
- Security metrics
- Audit logging

**Worker Service**:
- Service account authentication
- Secure environment variables
- API key management

### 3. Network

**Ingress**:
- API Service: Public (dengan authentication)
- Worker Service: Internal only (via Pub/Sub)

**Egress**:
- Firestore: Private Google network
- Cloud Storage: Private Google network
- External AI API: HTTPS only

---

## 🚀 Deployment Scenarios

### Scenario 1: Normal Development

```bash
# 1. Developer edit code
vim app.py

# 2. Test locally (optional)
python app.py

# 3. Commit & push
git add .
git commit -m "Fix bug"
git push origin main

# 4. Wait 5-10 minutes
# Cloud Build automatically deploys

# 5. Verify deployment
curl https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app/health
```

**Timeline**:
- Push: 0 min
- Build start: 0-1 min
- Build complete: 5-8 min
- Service updated: 8-10 min

---

### Scenario 2: Hotfix Production

```bash
# 1. Create hotfix branch (optional)
git checkout -b hotfix/critical-bug

# 2. Fix bug
vim app.py

# 3. Test locally
python app.py

# 4. Merge to main
git checkout main
git merge hotfix/critical-bug

# 5. Push
git push origin main

# 6. Monitor deployment
gcloud builds list --project=bni-prod-dma-bnimove-ai --limit=1

# 7. Verify fix
curl https://text-doc-api-service-v2-lh5pr6ewdq-et.a.run.app/
```

---

### Scenario 3: Rollback

Jika deployment baru bermasalah:

```bash
# Option 1: Rollback via Console
# 1. Go to Cloud Run service
# 2. Click "REVISIONS" tab
# 3. Select previous revision
# 4. Click "MANAGE TRAFFIC"
# 5. Route 100% to previous revision

# Option 2: Rollback via CLI
gcloud run services update-traffic text-doc-api-service-v2 \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai
```

---

### Scenario 4: Manual Deployment

Jika auto-deploy tidak jalan:

```bash
# 1. Build locally
cd text-doc-api-service
docker build -t asia-southeast2-docker.pkg.dev/bni-prod-dma-bnimove-ai/cloud-run-source-deploy/text-doc-api-service-v2:manual .

# 2. Push to Artifact Registry
docker push asia-southeast2-docker.pkg.dev/bni-prod-dma-bnimove-ai/cloud-run-source-deploy/text-doc-api-service-v2:manual

# 3. Deploy to Cloud Run
gcloud run deploy text-doc-api-service-v2 \
  --image=asia-southeast2-docker.pkg.dev/bni-prod-dma-bnimove-ai/cloud-run-source-deploy/text-doc-api-service-v2:manual \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai
```

---

## 📊 Monitoring & Logging

### View Build Logs

```bash
# List recent builds
gcloud builds list --project=bni-prod-dma-bnimove-ai --limit=5

# View specific build
gcloud builds log BUILD_ID --project=bni-prod-dma-bnimove-ai
```

### View Service Logs

```bash
# API Service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=text-doc-api-service-v2" \
  --limit=50 \
  --project=bni-prod-dma-bnimove-ai

# Worker Service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=text-doc-worker-service-v2" \
  --limit=50 \
  --project=bni-prod-dma-bnimove-ai
```

### View Metrics

**Console**:
- https://console.cloud.google.com/run?project=bni-prod-dma-bnimove-ai

**Metrics**:
- Request count
- Request latency
- Error rate
- CPU utilization
- Memory utilization
- Instance count

---

## 🛠️ Troubleshooting

### Build Failed

**Check**:
```bash
# View build logs
gcloud builds log BUILD_ID --project=bni-prod-dma-bnimove-ai

# Common issues:
# - Syntax error in code
# - Missing dependencies in requirements.txt
# - Dockerfile error
# - Organization policy violation
```

### Deployment Failed

**Check**:
```bash
# View service status
gcloud run services describe text-doc-api-service-v2 \
  --region=asia-southeast2 \
  --project=bni-prod-dma-bnimove-ai

# Common issues:
# - Image not found
# - Insufficient permissions
# - Port configuration wrong
# - Environment variables missing
```

### Service Not Responding

**Check**:
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=text-doc-api-service-v2 AND severity>=ERROR" \
  --limit=20 \
  --project=bni-prod-dma-bnimove-ai

# Common issues:
# - Application crash on startup
# - Database connection failed
# - API key invalid
# - Timeout too short
```

---

## 📝 Best Practices

### 1. Code Changes

- ✅ Test locally before push
- ✅ Use meaningful commit messages
- ✅ Small, incremental changes
- ✅ Review code before merge
- ❌ Don't push directly to main (use PR)

### 2. Deployment

- ✅ Monitor build progress
- ✅ Verify deployment success
- ✅ Test endpoints after deployment
- ✅ Keep previous revisions for rollback
- ❌ Don't deploy during peak hours

### 3. Monitoring

- ✅ Check logs regularly
- ✅ Set up alerts for errors
- ✅ Monitor resource usage
- ✅ Track deployment frequency
- ❌ Don't ignore warnings

---

## 🎯 Summary

### Initial Deployment Flow (Pertama Kali):
```
Manual Deploy → Cloud Build → Artifact Registry → 
CREATE Cloud Run Service → Service LIVE → Setup Trigger
```

### Subsequent Deployment Flow (Otomatis):
```
Code Change → Git Push → Cloud Build Trigger → Build Docker Image → 
Push to Artifact Registry → Deploy to Cloud Run → Service Updated
```

**Key Points**:
1. **Initial Setup**: Manual deploy pertama untuk create service
2. **Automatic**: Setelah setup, push ke GitHub = auto-deploy
3. **Fast**: 5-10 menit dari push sampai live
4. **Safe**: Rollback mudah jika ada masalah
5. **Compliant**: Semua di region asia-southeast2
6. **Monitored**: Logs dan metrics tersedia

**Time to Deploy**: 5-10 minutes  
**Downtime**: 0 seconds (zero-downtime deployment)  
**Rollback Time**: < 1 minute

### Deployment Checklist

**Untuk Service Baru**:
- [ ] Manual deploy pertama dengan `gcloud run deploy --source .`
- [ ] Verify service created dan URL generated
- [ ] Setup Cloud Build Trigger via Console
- [ ] Test auto-deploy dengan push dummy commit
- [ ] Configure environment variables jika perlu
- [ ] Setup monitoring dan alerts

**Untuk Update Service Existing**:
- [ ] Edit code di local
- [ ] Test locally (optional)
- [ ] Commit dengan message yang jelas
- [ ] Push ke GitHub main branch
- [ ] Monitor build progress di Cloud Build
- [ ] Verify deployment success
- [ ] Test endpoints
- [ ] Check logs untuk errors

---

**Sistem deployment sudah production-ready dan fully automated!** 🚀
