import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ============ CONFIGURATION ============
BALL_CLASSES = [
    'basketball',
    'billiard_ball',
    'bowling_ball',
    'football',
    'tennis_ball',
    'volleyball'
]

IMAGE_SIZE = (192, 192)  # Must match training size
MODEL_PATH = 'Ball_sport_classifier_old.h5'
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for more detections

# Colors for display (BGR format for OpenCV)
CLASS_COLORS = {
    'basketball': (0, 165, 255),      # Orange
    'billiard_ball': (255, 255, 255), # White
    'bowling_ball': (0, 0, 0),        # Black
    'football': (0, 165, 255),        # Orange
    'tennis_ball': (0, 255, 0),       # Green
    'volleyball': (255, 255, 0)       # Cyan
}

class BallImageDetector:
    def __init__(self):
        """Initialize image detector with model"""
        self.model = None
        print("🔄 Initializing Ball Detector...")
        self.load_model()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file not found: {MODEL_PATH}\n"
                    f"Please ensure '{MODEL_PATH}' exists in: {os.path.abspath(MODEL_PATH)}"
                )
            
            self.model = load_model(MODEL_PATH)
            
            # Validate model output shape matches BALL_CLASSES
            output_shape = self.model.output_shape
            num_classes = output_shape[-1]
            
            if num_classes != len(BALL_CLASSES):
                raise ValueError(
                    f"Model output shape mismatch!\n"
                    f"Model outputs {num_classes} classes, "
                    f"but BALL_CLASSES has {len(BALL_CLASSES)} entries."
                )
            
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            print(f"✓ Model expects input shape: {self.model.input_shape}")
            print(f"✓ Model outputs {num_classes} classes\n")
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            raise
    
    def preprocess_image(self, img_path):
        """Load and preprocess image for model prediction"""
        try:
            # Load image with target size matching training
            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            
            # Convert to array
            img_array = image.img_to_array(img)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalize to [0, 1] - matching training preprocessing
            img_array = img_array / 255.0
            
            # Validate preprocessing
            if np.isnan(img_array).any():
                raise ValueError("Image preprocessing resulted in NaN values")
            
            return img_array, img
            
        except Exception as e:
            print(f"✗ Image preprocessing error: {str(e)}")
            return None, None
    
    def predict_image(self, img_path, top_k=6):
        """Predict ball type from image with detailed confidence scores"""
        try:
            img_array, original_img = self.preprocess_image(img_path)
            
            if img_array is None:
                raise ValueError("Failed to preprocess image")
            
            # Make prediction
            if self.model is None:
                raise ValueError("Model is not loaded.")
            
            predictions = self.model.predict(img_array, verbose=0)
            prediction_scores = predictions[0]
            
            # Get top prediction
            predicted_class = np.argmax(prediction_scores)
            confidence = prediction_scores[predicted_class]
            
            # Validate predictions
            if np.isnan(confidence):
                raise ValueError("Model prediction resulted in NaN values")
            
            return {
                'class': predicted_class,
                'class_name': BALL_CLASSES[predicted_class],
                'confidence': confidence,
                'all_scores': prediction_scores,
                'original_image': original_img
            }
            
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
            return None
    
    def display_results(self, result, img_path):
        """Display prediction results with visualization"""
        if result is None:
            print("✗ Could not process image")
            return
        
        predicted_class = result['class']
        confidence = result['confidence']
        all_scores = result['all_scores']
        original_img = result['original_image']
        
        # Print detailed results
        print("\n" + "="*70)
        print(f"📊 PREDICTION RESULTS FOR: {os.path.basename(img_path)}")
        print("="*70)
        
        # Top prediction
        print(f"\n🎯 TOP PREDICTION:")
        print(f"   Ball Type: {BALL_CLASSES[predicted_class].upper()}")
        print(f"   Confidence: {confidence*100:.2f}%")
        
        # All predictions sorted by confidence
        print(f"\n📋 ALL PREDICTIONS (sorted by confidence):")
        print("-" * 70)
        print(f"{'Rank':<6} {'Ball Type':<20} {'Confidence':<15} {'Progress Bar':<25}")
        print("-" * 70)
        
        sorted_indices = np.argsort(all_scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            ball_type = BALL_CLASSES[idx]
            confidence_pct = all_scores[idx] * 100
            bar_length = int(confidence_pct / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            prefix = "→" if idx == predicted_class else " "
            print(f"{prefix}{rank:<5} {ball_type:<20} {confidence_pct:>6.2f}%{'':<8} [{bar}]")
        
        print("-" * 70)
        print(f"\n{'='*70}")
        
        # Visualize with matplotlib
        self.visualize_prediction(original_img, result, img_path)
    
    def annotate_image_with_detection(self, img_array, predicted_class, confidence, class_name):
        """Add detection annotations directly on the image"""
        import cv2
        
        # Convert PIL image to numpy array if needed
        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array.copy()
        
        height, width = img_cv.shape[:2]
        
        # Define colors (BGR format for OpenCV)
        colors = {
            'basketball': (0, 165, 255),      # Orange
            'billiard_ball': (255, 255, 255), # White
            'bowling_ball': (0, 0, 0),        # Black
            'football': (0, 165, 255),        # Orange
            'tennis_ball': (0, 255, 0),       # Green
            'volleyball': (255, 255, 0)       # Cyan
        }
        
        ball_color = colors.get(class_name, (0, 255, 0))
        
        # Draw semi-transparent overlay
        overlay = img_cv.copy()
        
        # Top banner
        cv2.rectangle(overlay, (0, 0), (width, int(height * 0.2)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)
        
        # Bottom info box
        cv2.rectangle(img_cv, (0, int(height * 0.85)), (width, height), (0, 0, 0), -1)
        overlay2 = img_cv.copy()
        cv2.rectangle(overlay2, (0, int(height * 0.85)), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, img_cv, 0.4, 0, img_cv)
        
        # Calculate font scale based on image size
        font_scale = max(0.8, min(width, height) / 300)
        thickness = max(2, int(font_scale))
        
        # Main detection text
        text_main = f"BALL DETECTED: {class_name.upper()}"
        text_size_main = cv2.getTextSize(text_main, cv2.FONT_HERSHEY_BOLD, font_scale * 1.2, thickness + 1)[0]
        text_x_main = (width - text_size_main[0]) // 2
        text_y_main = int(height * 0.12)
        
        # Draw text with outline
        cv2.putText(img_cv, text_main, (text_x_main, text_y_main), 
                   cv2.FONT_HERSHEY_BOLD, font_scale * 1.2, (0, 0, 0), thickness + 3)
        cv2.putText(img_cv, text_main, (text_x_main, text_y_main), 
                   cv2.FONT_HERSHEY_BOLD, font_scale * 1.2, ball_color, thickness)
        
        # Confidence text
        text_conf = f"Confidence: {confidence*100:.1f}%"
        text_size_conf = cv2.getTextSize(text_conf, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x_conf = (width - text_size_conf[0]) // 2
        text_y_conf = int(height * 0.18)
        
        cv2.putText(img_cv, text_conf, (text_x_conf, text_y_conf), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_cv, text_conf, (text_x_conf, text_y_conf), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Bottom info
        text_info = f"Detected: {class_name.upper()}"
        text_size_info = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x_info = (width - text_size_info[0]) // 2
        text_y_info = int(height * 0.93)
        
        cv2.putText(img_cv, text_info, (text_x_info, text_y_info), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_cv, text_info, (text_x_info, text_y_info), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Convert back to RGB for matplotlib
        img_annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        return img_annotated
    
    def visualize_prediction(self, img, result, img_path):
        """Create detailed visualization of prediction with annotations on image"""
        predicted_class = result['class']
        confidence = result['confidence']
        all_scores = result['all_scores']
        class_name = result['class_name']
        
        fig = plt.figure(figsize=(16, 6))
        
        # Annotate the image with detection results
        img_annotated = self.annotate_image_with_detection(img, predicted_class, confidence, class_name)
        
        # Display annotated image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img_annotated)
        ax1.set_title(f'Image with Detection Results', 
                      fontsize=14, fontweight='bold', color='green')
        ax1.axis('off')
        
        # Display confidence scores as bar chart
        ax2 = plt.subplot(1, 2, 2)
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_classes = [BALL_CLASSES[i] for i in sorted_indices]
        sorted_scores = [all_scores[i] * 100 for i in sorted_indices]
        
        colors = ['green' if i == predicted_class else 'skyblue' for i in sorted_indices]
        bars = ax2.barh(sorted_classes, sorted_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax2.text(score + 1, i, f'{score:.2f}%', va='center', fontweight='bold')
        
        ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Confidence Scores', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 105)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"prediction_{timestamp}.png"
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved as: {save_name}")
        
        plt.show()
    
    def batch_predict(self, image_dir):
        """Predict on all images in a directory"""
        print(f"\n🔄 Processing all images in: {image_dir}")
        print("="*70)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in os.listdir(image_dir) 
                      if Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"✗ No images found in {image_dir}")
            return
        
        results = []
        
        for idx, filename in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, filename)
            print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            
            result = self.predict_image(img_path)
            if result:
                results.append({
                    'filename': filename,
                    'class': result['class_name'],
                    'confidence': result['confidence'] * 100
                })
                print(f"  ✓ Detected: {result['class_name']} ({result['confidence']*100:.2f}%)")
            else:
                print(f"  ✗ Failed to process")
        
        # Summary
        print("\n" + "="*70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*70)
        
        for result in results:
            print(f"{result['filename']:<30} → {result['class']:<20} ({result['confidence']:.2f}%)")
        
        print("="*70)


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🏀 BALL SPORT DETECTOR - IMAGE TESTING 🏀")
    print("="*70 + "\n")
    
    try:
        detector = BallImageDetector()
        
        while True:
            print("\nOptions:")
            print("  1. Test a single image")
            print("  2. Test all images in a directory")
            print("  3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                img_path = input("Enter image path: ").strip().strip('"\'')
                
                if not os.path.exists(img_path):
                    print(f"✗ File not found: {img_path}")
                    continue
                
                result = detector.predict_image(img_path)
                detector.display_results(result, img_path)
                
            elif choice == '2':
                dir_path = input("Enter directory path: ").strip().strip('"\'')
                
                if not os.path.isdir(dir_path):
                    print(f"✗ Directory not found: {dir_path}")
                    continue
                
                detector.batch_predict(dir_path)
                
            elif choice == '3':
                print("\n✓ Exiting...")
                break
            
            else:
                print("✗ Invalid choice. Please enter 1, 2, or 3.")
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print(f"\nPlease make sure:")
        print(f"  1. The model file '{MODEL_PATH}' exists in the current directory")
        print(f"  2. You have trained the model using training_model.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
