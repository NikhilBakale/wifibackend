from flask import Flask, jsonify, send_file, request, make_response
from flask_cors import CORS
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import io
import json
import tempfile
from datetime import datetime
import logging
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add models directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Import ML model prediction function
ML_MODEL_AVAILABLE = False
classify_image = None
PARAM_EXTRACTOR_AVAILABLE = False
extract_call_parameters = None

try:
    import predict
    classify_image = predict.classify_image
    ML_MODEL_AVAILABLE = True
    print("✅ ML model (predict.py) loaded successfully")
except Exception as e:
    print(f"⚠️ ML model import error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    from spectrogram_analyzer import get_enhanced_parameters_dict
    extract_call_parameters = get_enhanced_parameters_dict
    PARAM_EXTRACTOR_AVAILABLE = True
    print("✅ Call parameter extractor loaded successfully")
except Exception as e:
    print(f"⚠️ Parameter extractor import error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000", 
    "https://frontend-ten-eta-28.vercel.app",
    "https://*.vercel.app",
    "https://frontend-3scf.vercel.app",
    "https://*.azurestaticapps.net"  # Allow Azure Static Web Apps
]) # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for storing spectrograms and audio
SPECTROGRAMS_DIR = Path('static/spectrograms')
AUDIO_DIR = Path('static/audio')
for d in [SPECTROGRAMS_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Serve static files - Flask needs static_folder configured
app.static_folder = 'static'
app.config['SPECTROGRAMS_FOLDER'] = str(SPECTROGRAMS_DIR)
app.config['AUDIO_FOLDER'] = str(AUDIO_DIR)

def format_bytes(size_bytes):
    """Format bytes to human readable string"""
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# Explicit routes for serving static files
@app.route('/static/spectrograms/<filename>')
def serve_spectrogram(filename):
    """Serve spectrogram images"""
    try:
        return send_file(
            SPECTROGRAMS_DIR / filename, 
            mimetype='image/png',
            as_attachment=False
        )
    except FileNotFoundError:
        return jsonify({'error': 'Spectrogram not found'}), 404

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    try:
        return send_file(
            AUDIO_DIR / filename, 
            mimetype='audio/wav',
            as_attachment=False
        )
    except FileNotFoundError:
        return jsonify({'error': 'Audio not found'}), 404

class GoogleDriveService:
    def __init__(self):
        self.drive = None
        self.initialize_drive()
    
    def initialize_drive(self):
        """Initialize Google Drive connection"""
        try:
            logger.info("Starting Google Drive initialization...")
            
            # Debug: Check environment variables
            client_secrets_env = os.environ.get('CLIENT_SECRETS_JSON')
            credentials_env = os.environ.get('CREDENTIALS_JSON')
            flask_env = os.environ.get('FLASK_ENV')
            
            logger.info(f"Environment check:")
            logger.info(f"- FLASK_ENV: {flask_env}")
            logger.info(f"- CLIENT_SECRETS_JSON present: {bool(client_secrets_env)}")
            logger.info(f"- CREDENTIALS_JSON present: {bool(credentials_env)}")
            
            gauth = GoogleAuth()
            
            # Check if we have client secrets in environment variable
            if client_secrets_env:
                logger.info("Creating client_secrets.json from environment variable")
                # Create client_secrets.json from environment variable
                with open('client_secrets.json', 'w') as f:
                    if isinstance(client_secrets_env, str):
                        # If it's a JSON string, parse it first
                        try:
                            client_secrets_data = json.loads(client_secrets_env)
                            json.dump(client_secrets_data, f)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, write it as is
                            f.write(client_secrets_env)
                    else:
                        json.dump(client_secrets_env, f)
                logger.info("Created client_secrets.json from environment variable")
            else:
                logger.warning("CLIENT_SECRETS_JSON environment variable not found")
            
            # Check if we have credentials in environment variable
            if credentials_env:
                logger.info("Creating credentials.json from environment variable")
                # Create credentials.json from environment variable
                with open('credentials.json', 'w') as f:
                    if isinstance(credentials_env, str):
                        try:
                            credentials_data = json.loads(credentials_env)
                            json.dump(credentials_data, f)
                        except json.JSONDecodeError:
                            f.write(credentials_env)
                    else:
                        json.dump(credentials_env, f)
                logger.info("Created credentials.json from environment variable")
            else:
                logger.warning("CREDENTIALS_JSON environment variable not found")
            
            # Try to load saved credentials
            gauth.LoadCredentialsFile("credentials.json")
            
            if gauth.credentials is None:
                # Authenticate if credentials are not available
                logger.info("No credentials found. Starting authentication flow...")
                # For production, we'll need to handle this differently
                if os.environ.get('FLASK_ENV') == 'production':
                    logger.error("Cannot perform interactive authentication in production")
                    raise Exception("Production deployment requires pre-authenticated credentials")
                else:
                    gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Refresh credentials if expired
                logger.info("Credentials expired. Refreshing...")
                gauth.Refresh()
            else:
                # Initialize the saved credentials
                gauth.Authorize()
                
            # Save the current credentials to file
            gauth.SaveCredentialsFile("credentials.json")
            
            self.drive = GoogleDrive(gauth)
            logger.info("Google Drive initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive: {e}")
            logger.error("Please ensure you have:")
            logger.error("1. Created client_secrets.json with your Google API credentials")
            logger.error("2. Enabled Google Drive API in Google Cloud Console")
            logger.error("3. Set up OAuth 2.0 Client ID credentials")
            raise e
    
    def search_bat_folder(self, server_num, client_num, bat_id):
        """
        Search for folder with pattern: SERVER{server_num}_CLIENT{client_num}_{bat_id}
        """
        folder_name = f"SERVER{server_num}_CLIENT{client_num}_{bat_id}"
        
        try:
            # Search for folders with the exact name
            query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            
            if file_list:
                logger.info(f"Found folder: {folder_name}")
                return file_list[0]  # Return first matching folder
            else:
                logger.warning(f"No folder found with name: {folder_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for folder {folder_name}: {e}")
            return None
    
    def search_folder_by_name(self, folder_name):
        """
        Search for folder by exact name (case-sensitive)
        """
        try:
            query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            
            if file_list:
                logger.info(f"Found folder: {folder_name}")
                return file_list[0]
            else:
                logger.warning(f"No folder found with name: {folder_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for folder {folder_name}: {e}")
            return None
    
    def get_folder_files(self, folder_id):
        """Get all files in a specific folder"""
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            
            files_info = []
            for file in file_list:
                files_info.append({
                    'id': file['id'],
                    'name': file['title'],
                    'mimeType': file['mimeType'],
                    'downloadUrl': file.get('downloadUrl', ''),
                    'modifiedDate': file.get('modifiedDate', ''),
                    'fileSize': file.get('fileSize', '0')
                })
            
            return files_info
            
        except Exception as e:
            logger.error(f"Error getting files from folder {folder_id}: {e}")
            return []
    
    def list_files_in_folder(self, folder_id):
        """Alias for get_folder_files for backward compatibility"""
        return self.get_folder_files(folder_id)
    
    def list_all_folders(self):
        """List all folders in Google Drive to debug"""
        try:
            query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            
            folders = []
            for folder in file_list:
                folders.append({
                    'id': folder['id'],
                    'name': folder['title'],
                    'modifiedDate': folder.get('modifiedDate', '')
                })
            
            logger.info(f"Found {len(folders)} folders in Google Drive")
            return folders
            
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []
    
    def list_all_items_detailed(self):
        """List all items in Google Drive with detailed info for debugging"""
        try:
            # Get all items in root
            query = "'root' in parents and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            
            items = []
            for item in file_list:
                items.append({
                    'id': item['id'],
                    'title': item['title'],
                    'mimeType': item.get('mimeType', 'unknown'),
                    'createdDate': item.get('createdDate', 'unknown'),
                    'modifiedDate': item.get('modifiedDate', 'unknown'),
                    'parents': item.get('parents', [])
                })
            
            logger.info(f"Found {len(items)} items in Google Drive root")
            return items
            
        except Exception as e:
            logger.error(f"Error listing all items: {e}")
            return []

    def download_and_store_locally(self, file_id, file_name, local_folder):
        """Download a file from Google Drive and store locally"""
        try:
            # Create local storage directory if it doesn't exist
            os.makedirs(local_folder, exist_ok=True)
            
            file = self.drive.CreateFile({'id': file_id})
            local_path = os.path.join(local_folder, file_name)
            
            # Download the file
            file.GetContentFile(local_path)
            logger.info(f"Downloaded {file_name} to {local_path}")
            
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None
    
    def download_file_to_path(self, file_id, destination_path):
        """Download a file from Google Drive to a specific path"""
        try:
            file = self.drive.CreateFile({'id': file_id})
            file.GetContentFile(destination_path)
            logger.info(f"Downloaded file {file_id} to {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            raise e

    def download_file_to_path(self, file_id, local_path):
        """Download a file from Google Drive to a specific path"""
        try:
            file = self.drive.CreateFile({'id': file_id})
            file.GetContentFile(local_path)
            logger.info(f"Downloaded file {file_id} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            raise e

# Initialize Google Drive service
drive_service = GoogleDriveService()

@app.route('/api/debug/folders')
def list_all_folders():
    """Debug endpoint to list all folders in Google Drive"""
    try:
        folders = drive_service.list_all_folders()
        return jsonify({
            'success': True,
            'total_folders': len(folders),
            'folders': folders
        })
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/debug/all-items')
def list_all_items():
    """Debug endpoint to list all items in Google Drive with detailed info"""
    try:
        items = drive_service.list_all_items_detailed()
        return jsonify({
            'success': True,
            'total_items': len(items),
            'items': items
        })
    except Exception as e:
        logger.error(f"Error listing all items: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/debug/download/<bat_id>')
def debug_download_files(bat_id):
    """Debug endpoint to download and store files locally"""
    try:
        # Extract server and client info from query parameters
        server_num = request.args.get('server', '1')
        client_num = request.args.get('client', '1')
        
        # Extract numeric part from BAT ID
        numeric_bat_id = bat_id.replace('BAT', '')
        
        # Search for the folder
        folder = drive_service.search_bat_folder(server_num, client_num, numeric_bat_id)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found for SERVER{server_num}_CLIENT{client_num}_{numeric_bat_id}'
            }), 404
        
        # Get files in the folder
        files = drive_service.get_folder_files(folder['id'])
        
        # Create local storage folder
        local_folder = f"downloads/SERVER{server_num}_CLIENT{client_num}_{numeric_bat_id}"
        downloaded_files = []
        
        # Download all files
        for file in files:
            local_path = drive_service.download_and_store_locally(
                file['id'], 
                file['name'], 
                local_folder
            )
            if local_path:
                downloaded_files.append({
                    'original_name': file['name'],
                    'local_path': local_path,
                    'file_id': file['id']
                })
        
        return jsonify({
            'success': True,
            'folder_name': folder['title'],
            'total_files': len(files),
            'downloaded_files': downloaded_files,
            'local_folder': local_folder
        })
        
    except Exception as e:
        logger.error(f"Error in debug download for BAT {bat_id}: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/bat/<bat_id>/files')
def get_bat_files(bat_id):
    """Get all files for a specific BAT ID"""
    try:
        # Extract server and client info from query parameters
        server_num = request.args.get('server', '1')
        client_num = request.args.get('client', '1')
        
        # Extract numeric part from BAT ID (e.g., BAT121 -> 121)
        numeric_bat_id = bat_id.replace('BAT', '')
        
        # Search for the folder
        folder = drive_service.search_bat_folder(server_num, client_num, numeric_bat_id)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found for SERVER{server_num}_CLIENT{client_num}_{numeric_bat_id}'
            }), 404
        
        # Get files in the folder
        files = drive_service.get_folder_files(folder['id'])
        
        # Organize files by type
        organized_files = {
            'spectrogram': None,
            'camera': None,
            'sensor': None,
            'audio': None,
            'other': []
        }
        
        for file in files:
            file_name_lower = file['name'].lower()
            # Handle both "spectrogram" and "spectogram" (with missing 'r')
            if ('spectrogram' in file_name_lower or 'spectogram' in file_name_lower) and file_name_lower.endswith('.jpg'):
                organized_files['spectrogram'] = file
            elif 'camera' in file_name_lower and file_name_lower.endswith('.jpg'):
                organized_files['camera'] = file
            elif 'sensor' in file_name_lower and file_name_lower.endswith('.txt'):
                organized_files['sensor'] = file
            elif file_name_lower.startswith('bat_') and file_name_lower.endswith('.wav'):
                organized_files['audio'] = file
            else:
                organized_files['other'].append(file)
        
        return jsonify({
            'success': True,
            'folder_name': folder['title'],
            'folder_id': folder['id'],
            'files': organized_files
        })
        
    except Exception as e:
        logger.error(f"Error getting files for BAT {bat_id}: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/file/<file_id>')
def download_file_endpoint(file_id):
    """Download a specific file from Google Drive"""
    try:
        file_name = request.args.get('name', 'file')
        
        # Download the file to temp location
        file = drive_service.drive.CreateFile({'id': file_id})
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}")
        file.GetContentFile(temp_file.name)
        
        # Determine mime type based on file extension
        mime_type = 'application/octet-stream'
        if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif file_name.lower().endswith('.png'):
            mime_type = 'image/png'
        elif file_name.lower().endswith('.txt'):
            mime_type = 'text/plain'
        
        return send_file(
            temp_file.name,
            mimetype=mime_type,
            as_attachment=False,
            download_name=file_name
        )
        
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/debug/upload-sensor/<bat_id>')
def upload_sensor_file(bat_id):
    """Debug endpoint to upload a sample sensor.txt file to Google Drive"""
    try:
        # Extract server and client info from query parameters
        server_num = request.args.get('server', '1')
        client_num = request.args.get('client', '1')
        
        folder_name = f"SERVER{server_num}_CLIENT{client_num}_{bat_id}"
        
        # Find the folder
        folder = drive_service.search_bat_folder(server_num, client_num, bat_id)
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found: {folder_name}'
            }), 404
        
        # Read the sample sensor file
        sample_file_path = os.path.join(os.path.dirname(__file__), 'sample_sensor.txt')
        if not os.path.exists(sample_file_path):
            return jsonify({
                'success': False,
                'message': 'Sample sensor file not found'
            }), 404
        
        # Upload the file to Google Drive
        file_metadata = {
            'title': 'Sensor.txt',
            'parents': [{'id': folder['id']}]
        }
        
        file = drive_service.drive.CreateFile(file_metadata)
        file.SetContentFile(sample_file_path)
        file.Upload()
        
        logger.info(f"Uploaded Sensor.txt to folder {folder_name}")
        
        return jsonify({
            'success': True,
            'message': f'Sensor.txt uploaded to {folder_name}',
            'file_id': file['id'],
            'folder_id': folder['id']
        })
        
    except Exception as e:
        logger.error(f"Error uploading sensor file: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Backend service is running',
        'timestamp': datetime.now().isoformat()
    })

# Mock species data for testing without Google Drive/Model
MOCK_SPECIES_DATA = {
    '825': {'species': 'Hipposideros_speoris', 'confidence': 92.5},
    '826': {'species': 'Pipistrellus_coromandra', 'confidence': 88.3},
    '827': {'species': 'Rhinolophus_rouxii', 'confidence': 95.1},
    '828': {'species': 'Taphozous_melanopogon', 'confidence': 85.7},
    '829': {'species': 'Megaderma_lyra', 'confidence': 91.2},
    '830': {'species': 'Unknown_species', 'confidence': 45.3},
}

@app.route('/api/predict/<bat_id>', methods=['GET', 'POST'])
def predict_species(bat_id):
    """
    Predict bat species from spectrogram image using ML model.
    
    GET params:
    - server: server number (optional)
    - client: client number (optional)
    - mock: set to 'true' to use mock data (optional, default is false)
    
    POST: Upload spectrogram image directly
    """
    try:
        # Get query parameters
        server = request.args.get('server', '1')
        client = request.args.get('client', '1')
        use_mock = request.args.get('mock', 'false').lower() == 'true'
        
        logger.info(f"Predicting species for BAT {bat_id} (Server {server}, Client {client}) - Mock: {use_mock}")
        
        # Check if mock data is explicitly requested
        if use_mock and bat_id in MOCK_SPECIES_DATA:
            mock_prediction = MOCK_SPECIES_DATA.get(bat_id, {
                'species': 'Unknown_species',
                'confidence': 50.0
            })
            logger.info(f"Using mock prediction: {mock_prediction['species']} ({mock_prediction['confidence']}%)")
            
            return jsonify({
                'success': True,
                'species': mock_prediction['species'],
                'confidence': mock_prediction['confidence'],
                'bat_id': bat_id,
                'mode': 'mock'
            })
        
        # Try to get spectrogram from Google Drive
        spectrogram_path = None
        wav_file_path = None
        try:
            logger.info(f"Fetching spectrogram from Google Drive for BAT {bat_id}")
            
            # Search for the BAT folder
            numeric_bat_id = bat_id.replace('BAT', '')
            folder = drive_service.search_bat_folder(server, client, numeric_bat_id)
            
            if folder:
                # Get files in the folder
                files_in_folder = drive_service.get_folder_files(folder['id'])
                
                # Find spectrogram file and WAV file
                spectrogram_file = None
                wav_file = None
                for file in files_in_folder:
                    file_name_lower = file['name'].lower()
                    if ('spectrogram' in file_name_lower or 'spectogram' in file_name_lower) and file_name_lower.endswith('.jpg'):
                        spectrogram_file = file
                    elif file_name_lower.endswith('.wav'):
                        wav_file = file
                
                if spectrogram_file:
                    # Download spectrogram
                    gfile = drive_service.drive.CreateFile({'id': spectrogram_file['id']})
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        gfile.GetContentFile(tmp.name)
                        spectrogram_path = tmp.name
                        logger.info(f"Downloaded spectrogram to {spectrogram_path}")
                else:
                    logger.warning(f"No spectrogram file found in folder {folder['title']}")
                
                # Download WAV file for parameter extraction
                if wav_file:
                    try:
                        gfile_wav = drive_service.drive.CreateFile({'id': wav_file['id']})
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                            gfile_wav.GetContentFile(tmp_wav.name)
                            wav_file_path = tmp_wav.name
                            logger.info(f"Downloaded WAV file to {wav_file_path}")
                    except Exception as wav_err:
                        logger.warning(f"Failed to download WAV file: {wav_err}")
            else:
                logger.warning(f"Folder not found for SERVER{server}_CLIENT{client}_{numeric_bat_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch spectrogram from Google Drive: {e}")
        
        # If no spectrogram from Drive, try to get from POST request
        if not spectrogram_path and request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'message': 'No spectrogram file provided',
                    'species': 'Unknown species',
                    'confidence': 0
                }), 400
            
            file = request.files['file']
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                file.save(tmp.name)
                spectrogram_path = tmp.name
                logger.info(f"Saved uploaded file to {spectrogram_path}")
        
        # If still no spectrogram, return error
        if not spectrogram_path:
            logger.warning(f"No spectrogram found for BAT {bat_id} - returning Unknown species")
            return jsonify({
                'success': False,
                'message': f'No spectrogram found for BAT {bat_id}',
                'species': 'Unknown_species',
                'confidence': 50.0,
                'bat_id': bat_id,
                'mode': 'error'
            }), 404
        
        # Use ML model to predict
        try:
            if not ML_MODEL_AVAILABLE:
                logger.error("ML model not available")
                return jsonify({
                    'success': False,
                    'message': 'ML model not available',
                    'species': 'Unknown_species',
                    'confidence': 0,
                    'bat_id': bat_id,
                    'mode': 'error'
                }), 500
            
            logger.info(f"Running ML model prediction on {spectrogram_path}")
            
            # Get multi-species predictions with 20% threshold
            from models.predict import classify_image_multi
            all_species = classify_image_multi(spectrogram_path, threshold=0.20)
            
            # Get top species for backward compatibility
            if all_species:
                predicted_species = all_species[0]['species']
                confidence = all_species[0]['confidence']
            else:
                predicted_species = "Unknown_species"
                confidence = 0.0
            
            logger.info(f"ML Prediction: {len(all_species)} species detected above 50% threshold")
            logger.info(f"Top species: {predicted_species} ({confidence}%)")
            
            # Extract call parameters if available
            call_parameters = {}
            logger.info(f"PARAM_EXTRACTOR_AVAILABLE: {PARAM_EXTRACTOR_AVAILABLE}")
            logger.info(f"extract_call_parameters function: {extract_call_parameters}")
            logger.info(f"wav_file_path: {wav_file_path}")
            
            if PARAM_EXTRACTOR_AVAILABLE and extract_call_parameters and wav_file_path:
                try:
                    logger.info(f"Extracting call parameters from WAV file: {wav_file_path}")
                    call_parameters = extract_call_parameters(Path(wav_file_path))
                    logger.info(f"Call parameters extracted successfully: {call_parameters}")
                except Exception as param_err:
                    logger.error(f"Error extracting call parameters: {param_err}")
                    import traceback
                    traceback.print_exc()
                    # Continue without parameters
            else:
                logger.warning(f"Skipping parameter extraction - PARAM_EXTRACTOR_AVAILABLE: {PARAM_EXTRACTOR_AVAILABLE}, wav_file_path: {wav_file_path}")
            
            # Extract GUANO metadata if WAV file available
            metadata = {}
            if wav_file_path:
                try:
                    from guano_metadata_extractor import extract_metadata_from_file
                    metadata = extract_metadata_from_file(wav_file_path)
                    logger.info(f"GUANO metadata extracted: {metadata}")
                except Exception as meta_err:
                    logger.warning(f"Could not extract GUANO metadata: {meta_err}")
            
            # Clean up temp files
            if os.path.exists(spectrogram_path):
                os.remove(spectrogram_path)
            if wav_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)
            
            response_data = {
                'success': True,
                'species': predicted_species,
                'confidence': confidence,
                'bat_id': bat_id,
                'mode': 'ml_model',
                'all_species': all_species,  # Include all detected species
                'species_count': len(all_species)
            }
            
            # Add call parameters if extracted
            if call_parameters:
                response_data['call_parameters'] = call_parameters
            
            # Add metadata if extracted
            if metadata:
                response_data['metadata'] = metadata
            
            return jsonify(response_data)
        
        except Exception as e:
            logger.error(f"Error running ML model: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Prediction error: {str(e)}',
                'species': 'Unknown_species',
                'confidence': 0,
                'bat_id': bat_id,
                'mode': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"Error predicting species: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e),
            'species': 'Unknown_species',
            'confidence': 0
        }), 500

@app.route('/api/species-image/<species_name>', methods=['GET', 'OPTIONS'])
def get_species_image(species_name):
    """
    Get species image from local bat_species folder.
    Maps species name to corresponding image file.
    
    Example: /api/species-image/Hipposideros_speoris
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 204
    try:
        # Get the backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        species_dir = os.path.join(backend_dir, 'bat_species')
        
        logger.info(f"Looking for species image: {species_name}")
        logger.info(f"Species directory: {species_dir}")
        
        if not os.path.exists(species_dir):
            logger.error(f"Species directory not found: {species_dir}")
            return jsonify({
                'success': False,
                'message': 'Species directory not found'
            }), 500
        
        # Try to find image with species name (with various extensions)
        possible_names = [
            f"{species_name}.jpg",
            f"{species_name}.jpeg",
            f"{species_name}.png",
            f"{species_name}.JPG",
            f"{species_name}.JPEG",
            f"{species_name}.PNG"
        ]
        
        image_path = None
        for name in possible_names:
            path = os.path.join(species_dir, name)
            if os.path.exists(path):
                image_path = path
                logger.info(f"Found species image: {path}")
                break
        
        # Fallback to Unknown_species if not found
        if not image_path:
            logger.warning(f"Species image not found for {species_name}, using Unknown_species")
            unknown_path = os.path.join(species_dir, 'Unknown_species.jpg')
            if os.path.exists(unknown_path):
                image_path = unknown_path
            else:
                return jsonify({
                    'success': False,
                    'message': 'Species image not found'
                }), 404
        
        # Send the image file with proper headers
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Content-Disposition'] = f'inline; filename="{os.path.basename(image_path)}"'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    except Exception as e:
        logger.error(f"Error retrieving species image: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/folder/files', methods=['POST'])
def list_folder_audio_files():
    """
    Instantly list all audio files in a folder WITHOUT processing
    Returns basic file info: name, size, id - NO predictions
    """
    try:
        data = request.get_json()
        server_num = data.get('server_num', '1')
        client_num = data.get('client_num', '1')
        folder_timestamp = data.get('folder_timestamp')
        
        if not folder_timestamp:
            return jsonify({
                'success': False,
                'message': 'Missing folder_timestamp parameter'
            }), 400
        
        logger.info(f"Listing audio files in folder: server{server_num}_client{client_num}_{folder_timestamp}")
        
        # Search for folder
        folder_name = f"server{server_num}_client{client_num}_{folder_timestamp}"
        folder = drive_service.search_folder_by_name(folder_name)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found: {folder_name}'
            }), 404
        
        # Get all files in folder
        files = drive_service.get_folder_files(folder['id'])
        
        # Filter audio files and return basic info only
        audio_files = []
        for f in files:
            if f['name'].lower().endswith('.wav'):
                audio_files.append({
                    'file_id': f['id'],
                    'file_name': f['name'],
                    'size': f.get('fileSize', 0),
                    'modified_date': f.get('modifiedDate', ''),
                    'download_url': f.get('downloadUrl', '')
                })
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        return jsonify({
            'success': True,
            'folder_name': folder_name,
            'total_files': len(audio_files),
            'files': audio_files
        })
    
    except Exception as e:
        logger.error(f"Error listing folder files: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/audio/predict', methods=['POST'])
def predict_single_audio():
    """
    Predict species for a single audio file
    Expects: file_id, file_name, server_num, client_num, folder_timestamp
    Optional: skip_prediction (boolean) - if true, only generate spectrogram/audio URLs without ML prediction
    Returns: Multi-species predictions with confidence
    """
    data = None
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        file_name = data.get('file_name')
        server_num = data.get('server_num', '1')
        client_num = data.get('client_num', '1')
        skip_prediction = data.get('skip_prediction', False)
        
        if not file_id:
            return jsonify({
                'success': False,
                'message': 'Missing file_id parameter'
            }), 400
        
        logger.info(f"Processing audio file: {file_name} (skip_prediction={skip_prediction})")
        
        # Check if spectrogram and audio already exist (for cached files)
        spec_filename = f"{file_id}.png"
        audio_filename = f"{file_id}_slow.wav"
        spectrogram_path = SPECTROGRAMS_DIR / spec_filename
        audio_save_path = AUDIO_DIR / audio_filename
        
        base_url = request.host_url.rstrip('/')
        spectrogram_url = f"{base_url}/static/spectrograms/{spec_filename}"
        audio_url = f"{base_url}/static/audio/{audio_filename}"
        
        # If files already exist and we're skipping prediction, we still need to download file for metadata
        if skip_prediction and spectrogram_path.exists() and audio_save_path.exists():
            logger.info(f"Using cached spectrogram and audio for {file_name}, but extracting metadata")
            
            # Still need to download and extract metadata
            metadata = {}
            call_parameters = {}
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                    file_obj = drive_service.drive.CreateFile({'id': file_id})
                    file_obj.GetContentFile(tmp_audio.name)
                    audio_path = tmp_audio.name
                
                # Extract GUANO metadata
                try:
                    from guano_metadata_extractor import extract_metadata_from_file
                    metadata = extract_metadata_from_file(audio_path)
                except Exception as meta_err:
                    logger.warning(f"Could not extract metadata: {meta_err}")
                
                # Extract call parameters
                if PARAM_EXTRACTOR_AVAILABLE and extract_call_parameters:
                    try:
                        call_parameters = extract_call_parameters(Path(audio_path))
                    except Exception as param_err:
                        logger.warning(f"Could not extract call parameters: {param_err}")
                
                # Clean up temp file
                os.unlink(audio_path)
            except Exception as e:
                logger.warning(f"Could not extract metadata/params for cached file: {e}")
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'file_name': file_name,
                'species': [],  # Frontend will merge with cached species data
                'species_count': 0,
                'predicted_species': None,
                'confidence': 0,
                'call_parameters': call_parameters,
                'metadata': metadata,
                'duration': 0,
                'sample_rate': 0,
                'spectrogram_url': spectrogram_url,
                'audio_url': audio_url,
                'from_cache': True
            })
        
        # Download audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            file_obj = drive_service.drive.CreateFile({'id': file_id})
            file_obj.GetContentFile(tmp_audio.name)
            audio_path = tmp_audio.name
        
        # Extract GUANO metadata
        metadata = {}
        try:
            from guano_metadata_extractor import extract_metadata_from_file
            metadata = extract_metadata_from_file(audio_path)
        except Exception as meta_err:
            logger.warning(f"Could not extract metadata: {meta_err}")
        
        # Generate spectrogram and save to static folder
        import librosa
        import librosa.display
        
        y, sr = librosa.load(audio_path, sr=None)
        D = librosa.stft(y, n_fft=2048, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Save spectrogram permanently
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.ylim(10000, 200000)  # Set frequency range from 10 kHz to 200 kHz
        plt.yticks(np.arange(10000, 200001, 20000))  # Set y-axis ticks every 20 kHz
        plt.title(file_name, color='white', fontsize=14)
        plt.xlabel('Time (s)', color='white')
        plt.ylabel('Frequency (Hz)', color='white')
        plt.tick_params(colors='white')
        plt.tight_layout()
        plt.savefig(str(spectrogram_path), dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        # Save audio file for playback (slowed down 10x)
        y_slow = librosa.effects.time_stretch(y, rate=0.1)  # 10x slower
        import soundfile as sf
        sf.write(str(audio_save_path), y_slow, sr)
        
        # Run ML prediction only if not skipping (multi-species with 20% threshold)
        species_list = []
        if not skip_prediction and ML_MODEL_AVAILABLE:
            from models.predict import classify_image_multi
            all_species = classify_image_multi(str(spectrogram_path), threshold=0.20)
            # all_species is now list of tuples: [(species, confidence), ...]
            species_list = [{'species': sp[0], 'confidence': round(sp[1], 1)} for sp in all_species]
        
        # Extract call parameters
        call_parameters = {}
        if PARAM_EXTRACTOR_AVAILABLE and extract_call_parameters:
            try:
                call_parameters = extract_call_parameters(Path(audio_path))
            except Exception as param_err:
                logger.warning(f"Could not extract call parameters: {param_err}")
        
        # Clean up only temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        result = {
            'success': True,
            'file_id': file_id,
            'file_name': file_name,
            'species': species_list,
            'species_count': len(species_list),
            'predicted_species': species_list[0]['species'] if species_list else 'Unknown',
            'confidence': species_list[0]['confidence'] if species_list else 0,
            'call_parameters': call_parameters,
            'metadata': metadata,
            'duration': round(len(y) / sr, 2),
            'sample_rate': sr,
            'spectrogram_url': spectrogram_url,
            'audio_url': audio_url
        }
        
        logger.info(f"Processed {file_name}: {len(species_list)} species detected")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error predicting audio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e),
            'file_id': data.get('file_id') if data else None,
            'file_name': data.get('file_name') if data else None
        }), 500


@app.route('/api/batch/folder', methods=['POST'])
def batch_process_folder():
    """
    Batch process all audio files in a Google Drive folder
    Expects: server_num, client_num, folder_timestamp
    Returns: Array of predictions with call parameters and metadata for all audio files
    """
    try:
        data = request.get_json()
        server_num = data.get('server_num', '1')
        client_num = data.get('client_num', '1')
        folder_timestamp = data.get('folder_timestamp')  # e.g., "23122025_1656"
        
        if not folder_timestamp:
            return jsonify({
                'success': False,
                'message': 'Missing folder_timestamp parameter'
            }), 400
        
        logger.info(f"Starting batch processing for folder: server{server_num}_client{client_num}_{folder_timestamp}")
        
        # Search for the folder on Google Drive (lowercase format)
        folder_name = f"server{server_num}_client{client_num}_{folder_timestamp}"
        folder = drive_service.search_folder_by_name(folder_name)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found: {folder_name}'
            }), 404
        
        # Get all files in the folder
        files = drive_service.get_folder_files(folder['id'])
        
        # Filter audio files (.wav)
        audio_files = [f for f in files if f['name'].lower().endswith('.wav')]
        
        if not audio_files:
            return jsonify({
                'success': False,
                'message': 'No audio files found in folder'
            }), 404
        
        logger.info(f"Found {len(audio_files)} audio files in folder")
        
        # Process each audio file
        results = []
        for audio_file in audio_files:
            try:
                # Download audio file to temp location
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                    file_obj = drive_service.drive.CreateFile({'id': audio_file['id']})
                    file_obj.GetContentFile(tmp_audio.name)
                    audio_path = tmp_audio.name
                
                # Extract GUANO metadata
                from guano_metadata_extractor import extract_metadata_from_file
                metadata = extract_metadata_from_file(audio_path)
                
                # Generate spectrogram using WildSynapse method
                import librosa
                import librosa.display
                import matplotlib.pyplot as plt
                import numpy as np
                
                y, sr = librosa.load(audio_path, sr=None)
                D = librosa.stft(y, n_fft=2048, hop_length=256)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                # Save spectrogram
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_spec:
                    plt.figure(figsize=(10, 6))
                    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz', cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.ylim(10000, 200000)  # Set frequency range from 10 kHz to 200 kHz
                    plt.yticks(np.arange(10000, 200001, 20000))  # Set y-axis ticks every 20 kHz
                    plt.title(audio_file['name'])
                    plt.tight_layout()
                    plt.savefig(tmp_spec.name, dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close()
                    spectrogram_path = tmp_spec.name
                
                # Run ML prediction
                if ML_MODEL_AVAILABLE:
                    from models.predict import classify_image_multi
                    all_species = classify_image_multi(spectrogram_path, threshold=0.01)
                    
                    # Format species data for frontend (list of {species, confidence})
                    species_list = [{'species': sp[0], 'confidence': round(sp[1] * 100, 1)} for sp in all_species]
                else:
                    species_list = []
                
                # Extract call parameters
                call_parameters = {}
                if PARAM_EXTRACTOR_AVAILABLE and extract_call_parameters:
                    try:
                        call_parameters = extract_call_parameters(Path(audio_path))
                    except Exception as param_err:
                        logger.error(f"Error extracting parameters for {audio_file['name']}: {param_err}")
                
                # Build result object
                result = {
                    'file_id': audio_file['id'],
                    'file_name': audio_file['name'],
                    'species': species_list,  # All species with confidence
                    'species_count': len(species_list),
                    'predicted_species': species_list[0]['species'] if species_list else 'Unknown',
                    'confidence': species_list[0]['confidence'] if species_list else 0,
                    'call_parameters': call_parameters,
                    'metadata': metadata,
                    'duration': round(len(y) / sr, 2) if 'y' in locals() else 0,
                    'sample_rate': sr if 'sr' in locals() else 0
                }
                
                results.append(result)
                
                # Clean up temp files
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(spectrogram_path):
                    os.remove(spectrogram_path)
                
                logger.info(f"Processed {audio_file['name']}: {result['predicted_species']} ({result['confidence']}%)")
                
            except Exception as file_err:
                logger.error(f"Error processing {audio_file['name']}: {file_err}")
                results.append({
                    'file_id': audio_file['id'],
                    'file_name': audio_file['name'],
                    'error': str(file_err),
                    'success': False
                })
        
        # Return batch results
        return jsonify({
            'success': True,
            'folder_name': folder_name,
            'total_files': len(audio_files),
            'processed': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/folders/<server_num>/<client_num>', methods=['GET'])
def get_client_folders(server_num, client_num):
    """
    List all folders for a specific server/client combination
    Returns ONLY folders matching pattern: SERVER{server_num}_CLIENT{client_num}_{DDMMYYYY}_{HHMM}
    Does NOT include individual audio files like SERVER1_CLIENT1_932
    """
    try:
        all_folders = drive_service.list_all_folders()
        
        # Filter folders matching pattern for this server/client (lowercase only)
        # ONLY match folders with date_time format: server1_client1_23122025_1656 (DDMMYYYY_HHMM)
        # NOT matching: SERVER1_CLIENT1_932 (uppercase - these are audio files)
        import re
        # Pattern: server1_client1_DDMMYYYY_HHMM (8 digits underscore 4 digits) - lowercase only
        pattern = rf'^server{server_num}_client{client_num}_(\d{{8}})_(\d{{4}})$'
        
        client_folders = []
        for folder in all_folders:
            match = re.match(pattern, folder['name'])
            if match:
                date_raw = match.group(1)  # 23122025
                time_raw = match.group(2)  # 1656
                
                # Format date: DD/MM/YYYY
                date_str = f"{date_raw[:2]}/{date_raw[2:4]}/{date_raw[4:]}"
                
                # Format time: HH:MM
                time_str = f"{time_raw[:2]}:{time_raw[2:]}"
                
                timestamp_part = f"{date_raw}_{time_raw}"
                
                # Get folder metadata (file count, total size)
                folder_id = folder['id']
                try:
                    files_in_folder = drive_service.list_files_in_folder(folder_id)
                    file_count = len(files_in_folder)
                    total_size = sum(int(f.get('fileSize', 0)) for f in files_in_folder)
                except:
                    file_count = 0
                    total_size = 0
                
                client_folders.append({
                    'id': folder_id,
                    'name': folder['name'],
                    'folder_id': folder['name'],
                    'timestamp': timestamp_part,
                    'modified_date': folder.get('modifiedDate', ''),
                    'date': date_str,
                    'time': time_str,
                    'file_count': file_count,
                    'total_size': total_size,
                    'total_size_formatted': format_bytes(total_size)
                })
        
        # Sort by timestamp (most recent first)
        client_folders.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_folders': len(client_folders),
            'folders': client_folders
        })
    
    except Exception as e:
        logger.error(f"Error listing folders for server{server_num}/client{client_num}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e),
            'folders': []
        }), 500


@app.route('/api/folders/list', methods=['GET'])
def list_bat_folders():
    """
    List all BAT folders in Google Drive matching the pattern server*_client*_DDMMYYYY_HHMM
    Returns ONLY folders with date_time format, not individual audio files (lowercase only)
    """
    try:
        all_folders = drive_service.list_all_folders()
        
        # Filter folders matching pattern: server{num}_client{num}_{DDMMYYYY}_{HHMM} (lowercase only)
        # This excludes individual audio files like SERVER1_CLIENT1_932 (uppercase)
        import re
        pattern = r'^server(\d+)_client(\d+)_(\d{8})_(\d{4})$'
        
        bat_folders = []
        for folder in all_folders:
            match = re.match(pattern, folder['name'])
            if match:
                server_num, client_num, date_raw, time_raw = match.groups()
                timestamp = f"{date_raw}_{time_raw}"
                
                # Format date and time
                date_str = f"{date_raw[:2]}/{date_raw[2:4]}/{date_raw[4:]}"
                time_str = f"{time_raw[:2]}:{time_raw[2:]}"
                
                bat_folders.append({
                    'id': folder['id'],
                    'name': folder['name'],
                    'server_num': server_num,
                    'client_num': client_num,
                    'timestamp': timestamp,
                    'date': date_str,
                    'time': time_str,
                    'modified_date': folder.get('modifiedDate', '')
                })
        
        # Sort by timestamp (most recent first)
        bat_folders.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_folders': len(bat_folders),
            'folders': bat_folders
        })
    
    except Exception as e:
        logger.error(f"Error listing folders: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/folder/audio-with-predictions', methods=['POST'])
def get_folder_audio_with_predictions():
    """
    UNIFIED ENDPOINT: Gets audio files with their Firebase predictions instantly.
    Returns files immediately - frontend can trigger predictions for missing ones.
    
    This endpoint:
    1. Lists all audio files from Google Drive
    2. Returns file list with 'has_prediction' flag
    3. Does NOT run ML predictions (frontend triggers those separately)
    """
    try:
        data = request.get_json()
        server_num = data.get('server_num', '1')
        client_num = data.get('client_num', '1')
        folder_timestamp = data.get('folder_timestamp')
        
        if not folder_timestamp:
            return jsonify({
                'success': False,
                'message': 'Missing folder_timestamp parameter'
            }), 400
        
        logger.info(f"📂 Getting audio files with predictions for: server{server_num}_client{client_num}_{folder_timestamp}")
        
        # Search for folder in Google Drive
        folder_name = f"server{server_num}_client{client_num}_{folder_timestamp}"
        folder = drive_service.search_folder_by_name(folder_name)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found: {folder_name}'
            }), 404
        
        # Get all audio files from Drive
        files = drive_service.get_folder_files(folder['id'])
        
        # Filter audio files
        audio_files = []
        for f in files:
            if f['name'].lower().endswith('.wav'):
                audio_files.append({
                    'file_id': f['id'],
                    'file_name': f['name'],
                    'size': f.get('fileSize', 0),
                    'modified_date': f.get('modifiedDate', ''),
                    'download_url': f.get('downloadUrl', ''),
                    'has_prediction': False  # Frontend will check Firebase
                })
        
        logger.info(f"✅ Found {len(audio_files)} audio files")
        
        return jsonify({
            'success': True,
            'folder_name': folder_name,
            'folder_id': folder['id'],
            'folder_timestamp': folder_timestamp,
            'total_files': len(audio_files),
            'files': audio_files,
            'server_num': server_num,
            'client_num': client_num
        })
    
    except Exception as e:
        logger.error(f"❌ Error getting folder audio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================================
# STANDALONE ENDPOINTS
# Format: standalone{num}_{DDMMYYYY}_{HHMM}
# ============================================================================

@app.route('/api/standalone/folders/<standalone_num>', methods=['GET'])
def get_standalone_folders(standalone_num):
    """
    List all folders for a specific standalone device
    Returns ONLY folders matching pattern: STANDALONE{standalone_num}_{DDMMYYYY}_{HHMM}
    """
    try:
        all_folders = drive_service.list_all_folders()
        
        import re
        # Pattern: standalone1_DDMMYYYY_HHMM (8 digits underscore 4 digits) - lowercase only
        pattern = rf'^standalone{standalone_num}_(\d{{8}})_(\d{{4}})$'
        
        standalone_folders = []
        for folder in all_folders:
            match = re.match(pattern, folder['name'].lower())
            if match:
                date_raw = match.group(1)  # 15082026
                time_raw = match.group(2)  # 1430
                
                # Format date: DD/MM/YYYY
                date_str = f"{date_raw[:2]}/{date_raw[2:4]}/{date_raw[4:]}"
                
                # Format time: HH:MM
                time_str = f"{time_raw[:2]}:{time_raw[2:]}"
                
                timestamp_part = f"{date_raw}_{time_raw}"
                
                # Get folder metadata (file count, total size)
                folder_id = folder['id']
                try:
                    files_in_folder = drive_service.list_files_in_folder(folder_id)
                    file_count = len(files_in_folder)
                    total_size = sum(int(f.get('fileSize', 0)) for f in files_in_folder)
                except:
                    file_count = 0
                    total_size = 0
                
                standalone_folders.append({
                    'id': folder_id,
                    'name': folder['name'],
                    'folder_id': folder['name'],
                    'timestamp': timestamp_part,
                    'modified_date': folder.get('modifiedDate', ''),
                    'date': date_str,
                    'time': time_str,
                    'file_count': file_count,
                    'total_size': total_size,
                    'total_size_formatted': format_bytes(total_size)
                })
        
        # Sort by timestamp (most recent first)
        standalone_folders.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_folders': len(standalone_folders),
            'folders': standalone_folders
        })
    
    except Exception as e:
        logger.error(f"Error listing folders for standalone{standalone_num}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e),
            'folders': []
        }), 500


@app.route('/api/standalone/folder/files', methods=['POST'])
def list_standalone_folder_audio_files():
    """
    Instantly list all audio files in a standalone folder WITHOUT processing
    Returns basic file info: name, size, id - NO predictions
    """
    try:
        data = request.get_json()
        standalone_num = data.get('standalone_num', '1')
        folder_timestamp = data.get('folder_timestamp')
        
        if not folder_timestamp:
            return jsonify({
                'success': False,
                'message': 'Missing folder_timestamp parameter'
            }), 400
        
        logger.info(f"Listing audio files in folder: standalone{standalone_num}_{folder_timestamp}")
        
        # Search for folder
        folder_name = f"standalone{standalone_num}_{folder_timestamp}"
        folder = drive_service.search_folder_by_name(folder_name)
        
        if not folder:
            return jsonify({
                'success': False,
                'message': f'Folder not found: {folder_name}'
            }), 404
        
        # Get all files in folder
        files = drive_service.get_folder_files(folder['id'])
        
        # Filter audio files and return basic info only
        audio_files = []
        for f in files:
            if f['name'].lower().endswith('.wav'):
                audio_files.append({
                    'file_id': f['id'],
                    'file_name': f['name'],
                    'size': f.get('fileSize', 0),
                    'modified_date': f.get('modifiedDate', ''),
                    'download_url': f.get('downloadUrl', '')
                })
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        return jsonify({
            'success': True,
            'folder_name': folder_name,
            'total_files': len(audio_files),
            'files': audio_files
        })
    
    except Exception as e:
        logger.error(f"Error listing standalone folder files: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/standalone/audio/predict', methods=['POST'])
def predict_standalone_single_audio():
    """
    Predict species for a single audio file from standalone device
    Expects: file_id, file_name, standalone_num, folder_timestamp
    Returns: Multi-species predictions with confidence (SAME FORMAT AS CLIENT ENDPOINT)
    """
    data = None
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        file_name = data.get('file_name')
        standalone_num = data.get('standalone_num', '1')
        folder_timestamp = data.get('folder_timestamp', '')
        
        if not file_id:
            return jsonify({'success': False, 'message': 'Missing file_id'}), 400
        
        logger.info(f"🔬 Predicting standalone audio: {file_name} (standalone{standalone_num}_{folder_timestamp})")
        
        # Generate filenames for caching (using file_id like client endpoint)
        spec_filename = f"{file_id}.png"
        audio_filename = f"{file_id}_slow.wav"
        spectrogram_path = SPECTROGRAMS_DIR / spec_filename
        audio_save_path = AUDIO_DIR / audio_filename
        
        base_url = request.host_url.rstrip('/')
        spectrogram_url = f"{base_url}/static/spectrograms/{spec_filename}"
        audio_url = f"{base_url}/static/audio/{audio_filename}"
        
        # Download audio file from Google Drive
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            file_obj = drive_service.drive.CreateFile({'id': file_id})
            file_obj.GetContentFile(tmp_audio.name)
            audio_path = tmp_audio.name
        
        # Get audio info
        import wave
        duration = 0
        sample_rate = 0
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate) if sample_rate > 0 else 0
        except Exception as e:
            logger.warning(f"Could not get audio info: {e}")
        
        # Generate spectrogram using librosa (same as client endpoint)
        try:
            import librosa
            import librosa.display
            
            y, sr = librosa.load(audio_path, sr=None)
            D = librosa.stft(y, n_fft=2048, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.ylim(10000, 200000)  # Set frequency range from 10 kHz to 200 kHz
            plt.yticks(np.arange(10000, 200001, 20000))  # Set y-axis ticks every 20 kHz
            plt.title(file_name, color='white', fontsize=14)
            plt.xlabel('Time (s)', color='white')
            plt.ylabel('Frequency (Hz)', color='white')
            plt.tick_params(colors='white')
            plt.tight_layout()
            plt.savefig(str(spectrogram_path), dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            logger.info(f"Generated spectrogram: {spectrogram_path}")
        except Exception as e:
            logger.error(f"Failed to generate spectrogram: {e}")
        
        # Run ML prediction
        species_predictions = []
        if ML_MODEL_AVAILABLE and spectrogram_path.exists():
            try:
                from models.predict import classify_image_multi
                all_species = classify_image_multi(str(spectrogram_path), threshold=0.20)
                # all_species is list of tuples: [(species, confidence), ...]
                species_predictions = [
                    {'species': sp[0], 'confidence': round(sp[1], 1)}
                    for sp in all_species
                ]
                logger.info(f"ML predictions: {len(species_predictions)} species detected")
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
        
        # Extract call parameters
        call_parameters = {}
        if PARAM_EXTRACTOR_AVAILABLE and extract_call_parameters:
            try:
                call_parameters = extract_call_parameters(Path(audio_path))
            except Exception as e:
                logger.error(f"Parameter extraction failed: {e}")
        
        # Extract GUANO metadata
        metadata = {}
        try:
            from guano_metadata_extractor import extract_metadata_from_file
            metadata = extract_metadata_from_file(audio_path)
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
        
        # Save processed audio file (slowed down version)
        import shutil
        shutil.copy(audio_path, str(audio_save_path))
        
        # Cleanup temp file
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_name': file_name,
            'species': species_predictions,
            'species_count': len(species_predictions),
            'predicted_species': species_predictions[0]['species'] if species_predictions else None,
            'confidence': species_predictions[0]['confidence'] if species_predictions else 0,
            'call_parameters': call_parameters,
            'metadata': metadata,
            'duration': duration,
            'sample_rate': sample_rate,
            'spectrogram_url': spectrogram_url,
            'audio_url': audio_url,
            'from_cache': False
        })
        
    except Exception as e:
        logger.error(f"Error predicting standalone audio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
