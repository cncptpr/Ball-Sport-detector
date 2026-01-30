import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

# ============ CONFIGURATION ============
BALL_CLASSES = [
    'basketball',
    'billiard_ball',
    'bowling_ball',
    'football',
    'tennis_ball',
    'volleyball'
]

IMAGE_SIZE = (192, 192)
MODEL_PATH = 'Ball_sport_classifier.h5'

# Colors for display (RGB)
COLORS = {
    'basketball': (255, 140, 0),
    'billiard_ball': (255, 255, 255),
    'bowling_ball': (50, 50, 50),
    'football': (139, 69, 19),
    'tennis_ball': (0, 255, 0),
    'volleyball': (255, 255, 0)
}


class SimpleBallDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Sport Detector")
        self.root.geometry("1200x900")
        self.root.config(bg='#2c3e50')
        
        self.model = None
        self.current_image = None
        self.prediction_result = None
        
        self.load_model()
        self.create_ui()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(MODEL_PATH):
                error_msg = f"Model file not found: {MODEL_PATH}\nCurrent directory: {os.getcwd()}"
                print(error_msg)
                messagebox.showerror("Error", error_msg)
                self.model = None
                return
            
            print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
            self.model = load_model(MODEL_PATH, compile=False)
            
            # Recompile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            
        except Exception as e:
            error_msg = f"Failed to load model:\n{str(e)}\n\nFull path: {os.path.abspath(MODEL_PATH)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.model = None
    
    def create_ui(self):
        """Create the user interface"""
        # Title
        title = tk.Label(
            self.root,
            text="🏀 Ball Sport Detector",
            font=("Arial", 28, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            self.root,
            text="📁 Upload Ball Image",
            command=self.upload_image,
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            padx=30,
            pady=15,
            cursor="hand2"
        )
        upload_btn.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Upload an image of a ball to identify it",
            font=("Arial", 12),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        self.status_label.pack(pady=5)
        
        # Image canvas with scrollbar
        canvas_frame = tk.Frame(self.root, bg='#34495e', relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#34495e',
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results frame with scrollable text
        results_frame = tk.LabelFrame(
            self.root,
            text="Detection Results",
            font=("Arial", 12, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        results_frame.pack(pady=10, padx=20, fill=tk.BOTH)
        
        # Results text with scrollbar
        text_scroll = tk.Scrollbar(results_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            results_frame,
            height=8,
            width=80,
            font=("Courier New", 11),
            bg='#34495e',
            fg='#ecf0f1',
            state=tk.DISABLED,
            yscrollcommand=text_scroll.set
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.results_text.yview)
    
    def upload_image(self):
        """Handle image upload and detection"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Ball Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.config(text="🔍 Analyzing ball...", fg='#3498db')
            self.root.update()
            
            # Load image
            self.current_image = Image.open(file_path).convert('RGB')
            
            # Detect ball type
            self.prediction_result = self.detect_ball(self.current_image)
            
            # Annotate and display image
            annotated_image = self.annotate_image(self.current_image.copy())
            self.display_image(annotated_image)
            
            # Show results
            self.show_results()
            
            ball_name = BALL_CLASSES[self.prediction_result['predicted_class']]
            confidence = self.prediction_result['confidence'] * 100
            
            self.status_label.config(
                text=f"✓ Detected: {ball_name.upper()} ({confidence:.1f}% confidence)",
                fg='#27ae60'
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            self.status_label.config(text="❌ Detection failed", fg='#e74c3c')
    
    def detect_ball(self, image):
        """Detect ball type from full image"""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        # Resize image to model input size
        img_resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Sort all predictions by confidence
        sorted_indices = np.argsort(predictions)[::-1]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions,
            'sorted_indices': sorted_indices
        }
    
    def annotate_image(self, image):
        """Add detection result overlay on image"""
        if self.prediction_result is None:
            return image
        
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        predicted_class = self.prediction_result['predicted_class']
        confidence = self.prediction_result['confidence']
        ball_name = BALL_CLASSES[predicted_class]
        color = COLORS.get(ball_name, (255, 0, 0))
        
        # Load font
        try:
            title_font = ImageFont.truetype("arial.ttf", size=max(40, int(height * 0.06)))
            subtitle_font = ImageFont.truetype("arial.ttf", size=max(30, int(height * 0.045)))
        except:
            try:
                title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=max(40, int(height * 0.06)))
                subtitle_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=max(30, int(height * 0.045)))
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Top banner
        banner_height = int(height * 0.25)
        overlay_draw.rectangle([(0, 0), (width, banner_height)], fill=(0, 0, 0, 200))
        
        # Bottom banner
        bottom_start = int(height * 0.88)
        overlay_draw.rectangle([(0, bottom_start), (width, height)], fill=(0, 0, 0, 200))
        
        # Blend overlay
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)
        image = image.convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw main title
        title_text = f"{ball_name.upper()}"
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = int(banner_height * 0.25)
        
        # Draw text with outline/shadow
        for offset in range(-3, 4):
            for offset2 in range(-3, 4):
                if offset != 0 or offset2 != 0:
                    draw.text((text_x + offset, text_y + offset2), title_text, 
                             font=title_font, fill=(0, 0, 0))
        draw.text((text_x, text_y), title_text, font=title_font, fill=color)
        
        # Draw confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        bbox = draw.textbbox((0, 0), conf_text, font=subtitle_font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = int(banner_height * 0.65)
        
        for offset in range(-2, 3):
            for offset2 in range(-2, 3):
                if offset != 0 or offset2 != 0:
                    draw.text((text_x + offset, text_y + offset2), conf_text, 
                             font=subtitle_font, fill=(0, 0, 0))
        draw.text((text_x, text_y), conf_text, font=subtitle_font, fill=(255, 255, 255))
        
        # Draw bottom info
        info_text = f"Ball Type Detected: {ball_name.upper()}"
        bbox = draw.textbbox((0, 0), info_text, font=subtitle_font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = bottom_start + int((height - bottom_start) * 0.3)
        
        for offset in range(-2, 3):
            for offset2 in range(-2, 3):
                if offset != 0 or offset2 != 0:
                    draw.text((text_x + offset, text_y + offset2), info_text, 
                             font=subtitle_font, fill=(0, 0, 0))
        draw.text((text_x, text_y), info_text, font=subtitle_font, fill=color)
        
        return image
    
    def display_image(self, image):
        """Display image resized to fit canvas while maintaining aspect ratio"""
        # Get canvas dimensions
        self.root.update()  # Update to get actual canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas not yet sized, use default
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 1100
            canvas_height = 500
        
        # Calculate scaling to fit image in canvas
        img_width, img_height = image.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
        
        # Resize image if needed
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Center image on canvas
        img_x = (canvas_width - image.width) // 2
        img_y = (canvas_height - image.height) // 2
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(img_x, img_y, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep reference
        
        # No scroll region needed since image fits
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
    
    def show_results(self):
        """Display detailed prediction results"""
        if self.prediction_result is None:
            return
        
        predicted_class = self.prediction_result['predicted_class']
        confidence = self.prediction_result['confidence']
        all_predictions = self.prediction_result['all_predictions']
        sorted_indices = self.prediction_result['sorted_indices']
        
        ball_name = BALL_CLASSES[predicted_class]
        
        # Build results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        result = "=" * 70 + "\n"
        result += "🎯 BALL IDENTIFICATION RESULTS\n"
        result += "=" * 70 + "\n\n"
        
        result += f"🏆 IDENTIFIED BALL TYPE: {ball_name.upper()}\n"
        result += f"📊 Confidence Score: {confidence*100:.2f}%\n"
        result += f"✓ Prediction Accuracy: {'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.5 else 'LOW'}\n\n"
        
        result += "=" * 70 + "\n"
        result += "📋 ALL PREDICTIONS (Ranked by Confidence):\n"
        result += "=" * 70 + "\n\n"
        
        result += f"{'Rank':<6} {'Ball Type':<20} {'Confidence':<12} {'Bar'}\n"
        result += "-" * 70 + "\n"
        
        for rank, idx in enumerate(sorted_indices, 1):
            ball_type = BALL_CLASSES[idx]
            conf = all_predictions[idx] * 100
            bar_length = int(conf / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            marker = "→" if idx == predicted_class else " "
            result += f"{marker} {rank:<4} {ball_type:<20} {conf:>6.2f}%    [{bar}]\n"
        
        result += "\n" + "=" * 70 + "\n"
        result += "💡 INTERPRETATION:\n"
        result += "=" * 70 + "\n"
        
        if confidence >= 0.8:
            result += "✓ Very High Confidence - The model is very certain about this prediction.\n"
        elif confidence >= 0.6:
            result += "✓ High Confidence - The model is quite confident about this prediction.\n"
        elif confidence >= 0.4:
            result += "⚠ Medium Confidence - The model has some uncertainty.\n"
        else:
            result += "⚠ Low Confidence - The model is uncertain. Try a clearer image.\n"
        
        # Show top 3 predictions
        result += "\nTop 3 Predictions:\n"
        for i, idx in enumerate(sorted_indices[:3], 1):
            result += f"  {i}. {BALL_CLASSES[idx]}: {all_predictions[idx]*100:.2f}%\n"
        
        result += "\n" + "=" * 70 + "\n"
        
        self.results_text.insert(1.0, result)
        self.results_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = SimpleBallDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()