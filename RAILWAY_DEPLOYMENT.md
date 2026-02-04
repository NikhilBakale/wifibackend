# Railway Deployment Guide

## Steps to Deploy Backend to Railway

### 1. Push your backend to GitHub
Make sure your backend folder is pushed to a GitHub repository.

### 2. Deploy to Railway
1. Go to [Railway.app](https://railway.app)
2. Sign up/Sign in with your GitHub account
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app

### 3. Configure Environment Variables
In Railway dashboard, go to your project → Variables tab and add:

**Required Variables:**
- `CLIENT_SECRETS_JSON`: Your entire client_secrets.json content as a string
- `FLASK_ENV`: Set to `production`
- `PORT`: Railway will set this automatically

**Optional Variables:**
- `PYTHONPATH`: Can help with imports if needed

### 4. Upload Google Drive Credentials
You have two options:

**Option A: Environment Variable (Recommended)**
1. Copy the entire content of your `client_secrets.json` file
2. Add it as the `CLIENT_SECRETS_JSON` environment variable in Railway
3. The app will automatically create the file on startup

**Option B: Manual Upload**
1. After deployment, you'll need to run the authentication flow once
2. Use Railway's console or logs to handle the Google OAuth flow

### 5. Test Deployment
After deployment:
1. Railway will provide a public URL (e.g., `https://your-app.railway.app`)
2. Test the health endpoint: `https://your-app.railway.app/api/health`
3. Test the folders endpoint: `https://your-app.railway.app/api/folders`

### 6. Update Frontend API URL
In your React app, update the API base URL to point to your Railway deployment.

## Important Notes:
- Make sure `client_secrets.json` is in your `.gitignore` to keep it secure
- The first deployment might take a few minutes
- Check Railway logs if there are any issues
- For Google Drive authentication in production, you may need to set up a service account instead of OAuth flow

## Troubleshooting:
- If deployment fails, check the build logs in Railway dashboard
- Ensure all dependencies are in `requirements.txt`
- Make sure `Procfile` is in the root of your backend folder