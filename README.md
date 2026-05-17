# CV_Project
3-D Reconstruction from images with features like single object segmentation and color and texture editing

## Steps to run the code on your local System after cloning it:
Steps to run:
1) python run_local.py --iterations 3000
2) (base) mantavya23311@LAPTOP-6013JAQC:~/vok-vision-main/backend/processor$ source venv/bin/activate
3)  a=1                               
  for f in *.jpg *.jpeg; do
  [ -e "$f" ] || continue
  printf -v num "%03d" $a
  mv -n "$f" "img_$num.jpg"
  ((a++))
  done

(base) mantavya23311@LAPTOP-6013JAQC:~/vok-vision-main/storage/uploads/test_project_001$


# VokVision: Professional 3D AI Reconstruction Engine

VokVision is an elite-grade 3D reconstruction platform that transforms a handful of 2D images into high-fidelity 3D Gaussian Splats. Engineered for professional workflows, it utilizes a state-of-the-art hybrid AI pipeline to deliver industrial-standard results.

---

## Elite Hybrid Architecture

Unlike basic reconstruction scripts, VokVision uses a Dual-Stage Hybrid Mapping strategy to ensure maximum fidelity:

1.  **VLM Image Audit**: Gemini 1.5 Flash filters out blurry or poorly lit images before processing begins to ensure data quality.
2.  **Hybrid Mapping (SfM)**: Uses original images (including background features) for perfect camera triangulation and stable poses.
3.  **Surgical Point-Cloud Masking**: Automatically removes background points using AI segmentation masks before the training phase.
4.  **Gaussian Splatting (30K)**: Trained on white-background datasets with 30,000 iterations for clean, professional, and floater-free 3D objects.

---

## Deployment Guide: Production Infrastructure

This section details the step-by-step process for transitioning VokVision from a local environment to a professional, cloud-native architecture. The plan is designed for maximum performance (NVIDIA 4090 GPUs) with minimized fixed costs.

### 1. Database and Orchestration Layer
The system requires a persistent database and a serverless message broker for job queuing.

*   **Database (MongoDB Atlas)**:
    1. Create a free shared cluster (M0) on [MongoDB Atlas](https://www.mongodb.com/atlas).
    2. Whitelist `0.0.0.0/0` or your specific server IP in the Network Access settings.
    3. Obtain the connection string (e.g., `mongodb+srv://user:password@cluster.mongodb.net/vokvision`).
*   **Message Broker (Upstash Redis)**:
    1. Create a serverless Redis instance on [Upstash](https://upstash.com).
    2. Obtain the `REDIS_URL` from the dashboard. This is critical for BullMQ job management.

### 2. High-Performance File Storage (Cloudflare R2)
Traditional storage like AWS S3 can be expensive due to egress fees for large 3D models. Cloudflare R2 is recommended for zero-egress cost.

1.  Create a bucket named `vokvision-storage` in [Cloudflare R2](https://www.cloudflare.com/products/r2/).
2.  Go to "Manage R2 API Tokens" and generate credentials with "Edit" permissions.
3.  Note down the `Access Key ID`, `Secret Access Key`, and the `S3 Endpoint URL`.
4.  **Code Change**: Update `backend/api/src/shared/middleware/upload.middleware.ts` to use the `aws-sdk` with your R2 endpoint to stream uploads directly to the cloud.

### 3. API API Backend Service (Render.com)
The Node.js backend serves as the "brain," managing users, projects, and the GPU pipeline.

1.  Create a "Web Service" on [Render](https://render.com).
2.  Connect your GitHub repository.
3.  **Build Command**: `cd backend/api && npm install && npm run build`
4.  **Start Command**: `cd backend/api && npm run start`
5.  **Environment Variables**:
    - `MONGODB_URI`: Your Atlas connection string.
    - `REDIS_URL`: Your Upstash URL.
    - `R2_ACCESS_KEY` / `R2_SECRET_KEY`: Your Cloudflare credentials.
    - `RUNPOD_API_KEY`: Required for triggering the GPU pipeline.

### 4. GPU Processing Pipeline (RunPod)
To provide professional-grade RTX 4090 processing without maintaining an expensive monthly server, RunPod Serverless is used.

1.  **Dockerization**: Package the `backend/processor` directory using the provided `Dockerfile`.
2.  **Push Image**: Build and push your image to Docker Hub or GitHub Container Registry.
3.  **RunPod Setup**:
    - Go to [RunPod Serverless](https://www.runpod.io/serverless-gpu).
    - Create a new Template using your Docker image.
    - Set the "Container Disk" to at least 20GB and "Volume Disk" to 40GB.
    - Deploy an Endpoint using an RTX 4090 (24GB VRAM) or RTX 6000 Ada (48GB VRAM).
4.  **Integration**: Update `processor.worker.ts` in the Node.js API to call your RunPod Endpoint URL instead of spawning a local Python process.

### 5. Mobile Application Deployment (Flutter)
The frontend must be reconfigured to point to your new cloud endpoint.

1.  **API URL**: Update the `_baseUrl` in `auth_repository.dart` and `project_repository.dart` to your Render service URL (e.g., `https://vokvision-api.onrender.com`).
2.  **Firebase**: Ensure your `google-services.json` (Android) and `GoogleService-Info.plist` (iOS) are correctly placed for production FCM notifications.
3.  **Build**:
    - Android: `flutter build apk --release`
    - iOS: `flutter build ipa --release`

---

## Local Development: Windows Setup

This section is for developers running the engine locally on a Windows machine with NVIDIA hardware.

### 1. System Requirements
- OS: Windows 10 or 11 (64-bit)
- GPU: NVIDIA RTX 30-series or 40-series (8GB+ VRAM recommended)
- RAM: 16GB minimum

### 2. Manual Prerequisites
- **CUDA Toolkit**: Install CUDA 11.8 or 12.1.
- **Python 3.10**: Ensure Python is added to the system PATH.
- **Git**: Required for repository management.
- **Redis**: Use WSL2 to run Redis or a native Windows port like Memurai.

### 3. Network Configuration
Both the mobile device and the development machine must be on the same Wi-Fi network.
- Update `LOCAL_IP` in `backend/api/.env`.
- Update the API URL in the Flutter repository files.

---

## Directory Structure

```plaintext
VokVision/
├── apps/mobile_app/        # Flutter Client (Dart)
├── backend/api/            # Node.js Orchestrator (TypeScript)
├── backend/processor/      # Python AI Engine (MASt3R, OpenSplat, VLM)
├── pipeline/               # Core AI System Modules
└── storage/                # Local data storage (Ignored by Git in production)
```

## Maintenance and Checkpoints

The system relies on large model checkpoints (e.g., MASt3R ViT-Large). 
- **Production**: These should be baked into your Docker image for fast startup.
- **Local**: These will download automatically on the first execution.

---

**Industry Recommendation**: For production-grade models, always ensure the input images have consistent exposure and at least 50% overlap between consecutive frames.
