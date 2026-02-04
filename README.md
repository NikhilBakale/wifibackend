# Google Drive Backend Service

This backend service handles fetching files from Google Drive folders with the naming pattern: `SERVER{x}_CLIENT{y}_{batId}`

## Setup Instructions

### 1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Google Drive API Setup:

#### Step 2a: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Drive API:
   - Go to "APIs & Services" → "Library"
   - Search for "Google Drive API"
   - Click "Enable"

#### Step 2b: Create OAuth 2.0 Credentials
1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client ID"
3. If prompted, configure the OAuth consent screen first
4. Choose "Desktop application" as application type
5. Give it a name (e.g., "BCIT Admin Panel")
6. Download the JSON file
7. Rename it to `client_secrets.json` and place it in the backend folder

### 3. First-time Authentication:
```bash
# Run the setup script to authenticate
python setup_drive.py
```

This will:
- Check if your client_secrets.json is valid
- Open a browser for Google authentication
- Save credentials for future use

### 4. Run the backend service:
```bash
python app.py
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/bat/{batId}/files?server={serverNum}&client={clientNum}` - Get files for a BAT ID
- `GET /api/file/{fileId}?name={fileName}` - Download a specific file

## Folder Structure Expected in Google Drive

```
SERVER1_CLIENT1_121/
├── Spectrogram.jpg
├── Camera.jpg
└── Sensor.txt
```

## Troubleshooting

### "Missing required setting client_config" Error
- Make sure `client_secrets.json` exists and is valid
- Run `python setup_drive.py` to verify setup

### "Access denied" or "Permission denied" errors
- Ensure the Google account has access to the folders
- Check that Google Drive API is enabled in your project

### "File not found" errors
- Verify folder naming matches: `SERVER{x}_CLIENT{y}_{batId}`
- Check that files exist with exact names: `Spectrogram.jpg`, `Camera.jpg`, `Sensor.txt`

## Notes

- Credentials are saved in `credentials.json` for future use
- The service expects specific file names (case-sensitive)
- Files are temporarily downloaded and streamed to the frontend
- Make sure the Google account has access to all required folders
