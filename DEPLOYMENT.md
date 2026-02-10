# Deployment Guide

This guide explains how to deploy the Enesy application using Docker and GitHub Actions.

## Overview

The application consists of two Docker containers:
- **Backend**: Python FastAPI server (port 8001)
- **Frontend**: Node.js React application (port 5006)

## Deployment Options

### Option 1: GitHub Container Registry + Manual Deployment

The GitHub Actions workflow (`docker-build.yml`) automatically builds and pushes Docker images to GitHub Container Registry (ghcr.io) when you push to `test-deployment` or `main` branches.

**Setup:**
1. Push your code - images will be built automatically
2. Pull images on your server:
   ```bash
   docker pull ghcr.io/amertzani/enesy-backend:latest
   docker pull ghcr.io/amertzani/enesy-frontend:latest
   ```
3. Run with docker-compose:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Option 2: Railway Deployment

Railway is a platform that automatically deploys from GitHub.

**Setup:**
1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Create a new project and connect your GitHub repository
3. Add the `RAILWAY_TOKEN` secret to your GitHub repository:
   - Go to Railway → Settings → Tokens → Create Token
   - Copy the token
   - Go to GitHub → Settings → Secrets → Actions → New repository secret
   - Name: `RAILWAY_TOKEN`, Value: (paste token)
4. Railway will automatically detect `docker-compose.yml` and deploy both services

**Manual Railway CLI deployment:**
```bash
railway login
railway init
railway up
```

### Option 3: Render Deployment

Render provides free tier hosting for Docker containers.

**Setup:**
1. Go to [render.com](https://render.com) and sign in with GitHub
2. Create two Web Services:
   - **Backend Service:**
     - Name: `enesy-backend`
     - Environment: Docker
     - Dockerfile Path: `Dockerfile.backend`
     - Root Directory: `.`
     - Port: `8001`
   - **Frontend Service:**
     - Name: `enesy-frontend`
     - Environment: Docker
     - Dockerfile Path: `RandDKnowledgeGraph/Dockerfile.frontend`
     - Root Directory: `RandDKnowledgeGraph`
     - Port: `5006`
     - Environment Variable: `VITE_API_URL=https://enesy-backend.onrender.com`
3. Add GitHub secrets:
   - `RENDER_API_KEY`: Get from Render Dashboard → Account Settings → API Keys
   - `RENDER_BACKEND_SERVICE_ID`: Found in your backend service URL
   - `RENDER_FRONTEND_SERVICE_ID`: Found in your frontend service URL

### Option 4: Fly.io Deployment

Fly.io provides global Docker deployment.

**Setup:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Create apps:
   ```bash
   fly launch --name enesy-backend --dockerfile Dockerfile.backend
   fly launch --name enesy-frontend --dockerfile RandDKnowledgeGraph/Dockerfile.frontend
   ```
4. Configure networking and environment variables in Fly dashboard

## Environment Variables

### Backend
- `API_PORT`: Port for backend server (default: 8001)
- `API_HOST`: Host to bind to (default: 0.0.0.0)

### Frontend
- `VITE_API_URL`: Backend API URL (default: http://localhost:8001)
- `NODE_ENV`: Set to `production` for production builds
- `PORT`: Port for frontend server (default: 5006)

## Production Considerations

1. **API URL Configuration**: Update `VITE_API_URL` in frontend to point to your deployed backend URL
2. **CORS**: Backend allows all origins (`allow_origins=["*"]`). For production, restrict this to your frontend domain
3. **HTTPS**: Use a reverse proxy (nginx, Caddy) or platform-provided SSL for HTTPS
4. **Database**: If using persistent storage, configure volume mounts in docker-compose
5. **Secrets**: Store sensitive data in environment variables or secrets management

## Testing Deployment

After deployment, verify:
- Backend health: `curl https://your-backend-url/api/documents`
- Frontend: Open `https://your-frontend-url` in browser
- API docs: `https://your-backend-url/docs`

## Troubleshooting

- **Images not building**: Check GitHub Actions logs
- **Services not connecting**: Verify `VITE_API_URL` points to correct backend URL
- **Port conflicts**: Ensure ports 8001 and 5006 are available
- **Build failures**: Check Dockerfile syntax and dependencies

## Current Status

✅ Docker files configured
✅ GitHub Actions workflows created
✅ Production docker-compose file ready
⏳ Choose deployment platform and configure secrets
