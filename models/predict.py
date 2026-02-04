import os, json, argparse

# Lazy imports - only import when needed to avoid numpy issues
_model = None
_device = None
_transform = None
_classes = None

def load_dependencies():
    """Lazy load all ML dependencies when first needed"""
    global _model, _device, _transform, _classes, torch, torch_nn, transforms, EfficientNet, cv2, np
    
    if _model is not None:
        return  # Already loaded
    
    # Import numpy FIRST to ensure it's available
    import numpy as np
    # Make sure numpy array operations work
    np.array([1, 2, 3])  # Test numpy
    
    # Now import torch and torchvision
    import torch
    import torch.nn as torch_nn
    from torchvision import transforms
    from efficientnet_pytorch import EfficientNet
    import cv2
    
    # Config - use absolute paths relative to this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "bat_28.pth")
    CLASSES_PATH = os.path.join(SCRIPT_DIR, "classes_28.json")
    
    # Load classes
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        _classes = json.load(f)
    
    # Device & transforms
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load model
    def load_model(model_path, num_classes):
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = model._fc.in_features
        model._fc = torch_nn.Sequential(torch_nn.Linear(num_ftrs, num_classes))
        
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval().to(_device)
        return model
    
    _model = load_model(MODEL_PATH, len(_classes))

def classify_image(img_path, threshold=0.01):
    """
    Classify an image using multi-label classification and return top species.
    Returns the top predicted species and its confidence.
    
    Args:
        img_path: Path to spectrogram image
        threshold: Minimum confidence threshold (0-1)
    
    Returns:
        (species_name, confidence_percentage)
    """
    # Lazy load dependencies on first call
    load_dependencies()
    
    # Use cv2 to load image (more compatible with numpy than PIL)
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Convert BGR to RGB (cv2 loads in BGR by default)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL for torchvision transforms
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img_rgb)
    
    # Apply transforms
    x = _transform(img_pil).unsqueeze(0).to(_device)
   
    # Get prediction using sigmoid for multi-label
    with torch.no_grad():
        logits = _model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get all detections above threshold
    detections = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            detections.append({
                'species': _classes[i],
                'confidence': float(prob * 100)
            })
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Return top prediction
    if detections:
        top = detections[0]
        species = top['species'].replace(" ", "_")
        return species, round(top['confidence'], 2)
    
    return "Unknown_species", 0.0

def classify_image_multi(img_path, threshold=0.01):
    """
    Classify an image and return multiple species predictions.
    
    Args:
        img_path: Path to spectrogram image
        threshold: Minimum confidence threshold (0-1)
    
    Returns:
        List of (species_name, confidence_percentage) tuples
    """
    # Lazy load dependencies on first call
    load_dependencies()
    
    # Use cv2 to load image
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img_rgb)
    
    x = _transform(img_pil).unsqueeze(0).to(_device)
   
    with torch.no_grad():
        logits = _model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    detections = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            detections.append((_classes[i].replace(" ", "_"), float(prob * 100)))
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    return detections

# --- CLI Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a bat spectrogram image using EfficientNet.")
    parser.add_argument('image_path', type=str, help='Path to the spectrogram image file (e.g., spectrogram.jpg)')
    
    args = parser.parse_args()
    
    prediction, confidence = classify_image(args.image_path)
    
    print("\n--- Final Result ---")
    print(f"Predicted Species: {prediction}")
    print(f"Confidence: {confidence}%")
    