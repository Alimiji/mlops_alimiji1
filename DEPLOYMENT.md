# Deployment Guide - WeatherPredict Pro

This guide explains how to deploy the WeatherPredict Pro MLOps pipeline with:
- **API** on Render (FastAPI backend)
- **UI** on Streamlit Cloud (Streamlit frontend)

---

## Architecture Overview

```
[GitHub Repository]
       |
       v
[GitHub Actions CI/CD]
       |
       +---> [Render] --> FastAPI API (https://weather-prediction-api.onrender.com)
       |
       +---> [Streamlit Cloud] --> Streamlit UI (https://your-app.streamlit.app)
                                        |
                                        v
                              [Calls API via HTTPS]
```

---

## Step 1: Deploy API on Render

### 1.1 Create Render Account
1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub account

### 1.2 Create New Web Service
1. Click **New** > **Web Service**
2. Connect your GitHub repository: `mlops_alimiji1`
3. Configure the service:
   - **Name**: `weather-prediction-api`
   - **Region**: Frankfurt (EU) or closest to you
   - **Branch**: `master`
   - **Runtime**: Python 3
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or Starter for production)

### 1.3 Set Environment Variables
In Render dashboard > Environment:
```
PYTHONPATH=/opt/render/project/src
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 1.4 Verify Deployment
After deployment, test your API:
```bash
# Replace with your actual Render URL
curl https://weather-prediction-api.onrender.com/health
curl https://weather-prediction-api.onrender.com/metrics
```

### 1.5 Get Render API Keys (for GitHub Actions)
1. Go to **Account Settings** > **API Keys**
2. Create a new API key
3. Go to your service **Settings** > copy the **Service ID**
4. Add these as GitHub Secrets:
   - `RENDER_API_KEY`: Your API key
   - `RENDER_SERVICE_ID`: Your service ID (starts with `srv-`)

---

## Step 2: Deploy UI on Streamlit Cloud

### 2.1 Create Streamlit Cloud Account
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub

### 2.2 Deploy the App
1. Click **New app**
2. Select your repository: `mlops_alimiji1`
3. Configure:
   - **Branch**: `master`
   - **Main file path**: `streamlit_app/app.py`
   - **App URL**: Choose a custom URL (e.g., `weatherpredict-pro`)

### 2.3 Configure Secrets
1. Go to your app > **Settings** > **Secrets**
2. Add the following secret (replace with your Render API URL):
```toml
API_URL = "https://weather-prediction-api.onrender.com"
```

### 2.4 Verify Deployment
1. Open your Streamlit app URL
2. Check the **Monitoring** page to verify API connection
3. Make a test prediction on the **Predictions** page

---

## Step 3: Configure GitHub Actions Secrets

Add these secrets in GitHub: **Settings** > **Secrets and variables** > **Actions**

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `RENDER_API_KEY` | Render API key | `rnd_xxxxxxxxxxxx` |
| `RENDER_SERVICE_ID` | Render service ID | `srv-xxxxxxxxxxxx` |
| `GDRIVE_CREDENTIALS_DATA` | Google Drive credentials (for DVC) | `{"type":"service_account",...}` |

---

## Step 4: Test the Full Pipeline

### 4.1 Trigger a Deployment
```bash
# Make a change and push
git add .
git commit -m "Test deployment"
git push origin master
```

### 4.2 Monitor GitHub Actions
1. Go to **Actions** tab in your GitHub repository
2. Watch the **CI Pipeline** run
3. Watch the **CD - Deploy API** run
4. Verify Render deployment triggered

### 4.3 Verify End-to-End
1. Open Streamlit app
2. Go to **Predictions** page
3. Enter weather features
4. Verify prediction is returned from Render API

---

## Troubleshooting

### API Not Responding on Render
- Check Render logs in the dashboard
- Verify PYTHONPATH is set correctly
- Ensure model files are in the repository

### Streamlit Can't Connect to API
- Verify `API_URL` secret is set correctly in Streamlit Cloud
- Check CORS is enabled in the API
- Test the API URL directly in browser

### GitHub Actions Failing
- Check workflow logs for errors
- Verify all secrets are configured
- Ensure branch name matches (master vs main)

---

## Local Development

### Run API Locally
```bash
make run-api
# or
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Streamlit Locally
```bash
make streamlit
# or
cd streamlit_app && streamlit run app.py
```

### Run Both with Docker
```bash
docker-compose up -d api streamlit
```

---

## Monitoring & Maintenance

### Health Checks
- API: `https://your-api.onrender.com/health`
- Metrics: `https://your-api.onrender.com/metrics`

### Logs
- **Render**: Dashboard > Logs
- **Streamlit Cloud**: App > Manage app > Logs

### Redeploy
- **Automatic**: Push to master branch
- **Manual**: GitHub Actions > CD - Deploy API > Run workflow

---

## Cost Considerations

| Service | Free Tier | Paid Options |
|---------|-----------|--------------|
| Render | 750 hours/month, spins down after inactivity | $7/month Starter |
| Streamlit Cloud | Unlimited for public apps | Private apps require Team plan |
| GitHub Actions | 2000 minutes/month | Pay per minute after |

---

## Next Steps

1. Set up custom domain for your API
2. Add authentication (API keys, JWT)
3. Configure monitoring alerts
4. Set up staging environment