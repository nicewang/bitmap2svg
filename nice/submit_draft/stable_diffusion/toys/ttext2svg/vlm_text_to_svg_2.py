import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO

class EnhancedVisionLanguageTextToSVG:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)
        
        # Initialize CLIP for semantic understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # SVG generation parameters (from C++ logic)
        self.MAX_SVG_SIZE_BYTES = 10000
        self.SVG_SIZE_SAFETY_MARGIN = 1000
        
        # Enhanced semantic categories
        self.semantic_categories = [
            "sky", "cloud", "mountain", "tree", "grass", "water", "building", 
            "person", "animal", "flower", "rock", "sun", "moon", "star",
            "road", "field", "forest", "lake", "river", "hill"
        ]
        
    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate image from text prompt using Stable Diffusion"""
        default_kwargs = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512
        }
        default_kwargs.update(kwargs)
        
        with torch.no_grad():
            result = self.pipe(prompt, **default_kwargs)
        
        return result.images[0]
    
    def compress_hex_color(self, r: int, g: int, b: int) -> str:
        """Compress hex color representation (from C++ logic)"""
        # Check if we can use 3-digit hex (like #RGB instead of #RRGGBB)
        if (r % 17 == 0 and g % 17 == 0 and b % 17 == 0 and 
            r // 17 < 16 and g // 17 < 16 and b // 17 < 16):
            return f"#{r//17:x}{g//17:x}{b//17:x}"
        else:
            return f"#{r:02x}{g:02x}{b:02x}"
    
    def perform_color_quantization(self, image: np.ndarray, num_colors: int = 8) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Enhanced color quantization using K-means (based on C++ logic)"""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with 3 channels")
        
        # Adaptive color count based on image size
        if num_colors <= 0:
            pixel_count = image.shape[0] * image.shape[1]
            if pixel_count < 16384:  # < 128x128
                num_colors = 6
            elif pixel_count < 65536:  # < 256x256
                num_colors = 8
            elif pixel_count < 262144:  # < 512x512
                num_colors = 10
            else:
                num_colors = 12
        
        # Ensure reasonable range
        num_colors = max(1, min(num_colors, 16))
        
        # Reshape image for K-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Handle edge case where we have fewer unique colors than requested
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) < num_colors:
            num_colors = len(unique_colors)
        
        # Perform K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        attempts = 5 if num_colors <= 8 else 7
        
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
        )
        
        # Create quantized image
        quantized_pixels = centers[labels.flatten()]
        quantized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)
        
        # Extract palette
        palette = [(int(c[0]), int(c[1]), int(c[2])) for c in centers]
        
        return quantized_image, palette
    
    def extract_contours_for_color(self, quantized_image: np.ndarray, target_color: Tuple[int, int, int], 
                                  min_contour_area: float = 100, 
                                  simplification_epsilon_factor: float = 0.02) -> List[np.ndarray]:
        """Extract and simplify contours for a specific color (from C++ logic)"""
        # Create mask for target color
        mask = cv2.inRange(quantized_image, target_color, target_color)
        
        if cv2.countNonZero(mask) == 0:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        simplified_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            # Simplify contour
            epsilon = max(0.5, simplification_epsilon_factor * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) >= 3:
                simplified_contours.append(simplified.reshape(-1, 2))
        
        return simplified_contours
    
    def calculate_feature_importance(self, contour: np.ndarray, area: float, 
                                   image_center: Tuple[float, float], 
                                   max_dist_from_center: float) -> float:
        """Calculate feature importance based on C++ logic"""
        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] > 1e-5:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
        
        # Distance from image center
        dist_from_center = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
        normalized_dist = dist_from_center / max_dist_from_center if max_dist_from_center > 1e-5 else 0
        
        # Importance calculation
        importance = area * (1.0 - normalized_dist) * (1.0 / (len(contour) + 1.0))
        
        return importance
    
    def classify_segments_enhanced(self, image: Image.Image, quantized_image: np.ndarray, 
                                 palette: List[Tuple[int, int, int]]) -> List[Dict]:
        """Enhanced segment classification using CLIP"""
        segment_info = []
        
        for color in palette:
            # Create mask for this color
            mask = cv2.inRange(quantized_image, color, color)
            
            if cv2.countNonZero(mask) < 1000:  # Skip small segments
                continue
            
            # Create masked image for CLIP
            img_array = np.array(image)
            masked_img = img_array.copy()
            
            # Apply mask
            mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
            masked_img = (masked_img * mask_3d).astype(np.uint8)
            
            # Convert to PIL
            pil_segment = Image.fromarray(masked_img)
            
            # Use CLIP to classify
            inputs = self.clip_processor(
                text=self.semantic_categories,
                images=pil_segment,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top prediction
            top_prob, top_idx = torch.max(probs, dim=1)
            predicted_category = self.semantic_categories[top_idx.item()]
            confidence = top_prob.item()
            
            segment_info.append({
                'color': color,
                'category': predicted_category,
                'confidence': confidence,
                'area': cv2.countNonZero(mask)
            })
        
        return segment_info
    
    def get_category_style_enhanced(self, category: str) -> Dict[str, str]:
        """Enhanced category styling"""
        style_map = {
            "sky": {"fill": "#87CEEB", "stroke": "none"},
            "cloud": {"fill": "#FFFFFF", "stroke": "#CCCCCC", "stroke-width": "0.5"},
            "mountain": {"fill": "#8B7355", "stroke": "#654321", "stroke-width": "1"},
            "tree": {"fill": "#228B22", "stroke": "#006400", "stroke-width": "0.5"},
            "grass": {"fill": "#9ACD32", "stroke": "#6B8E23", "stroke-width": "0.5"},
            "water": {"fill": "#4682B4", "stroke": "#191970", "stroke-width": "0.5"},
            "building": {"fill": "#D2691E", "stroke": "#8B4513", "stroke-width": "1"},
            "person": {"fill": "#FDBCB4", "stroke": "#8B4513", "stroke-width": "1"},
            "animal": {"fill": "#DEB887", "stroke": "#8B7355", "stroke-width": "1"},
            "flower": {"fill": "#FF69B4", "stroke": "#DC143C", "stroke-width": "0.5"},
            "rock": {"fill": "#708090", "stroke": "#2F4F4F", "stroke-width": "0.5"},
            "sun": {"fill": "#FFD700", "stroke": "#FFA500", "stroke-width": "1"},
            "moon": {"fill": "#F0E68C", "stroke": "#BDB76B", "stroke-width": "0.5"},
            "star": {"fill": "#FFFF00", "stroke": "#FFD700", "stroke-width": "0.5"},
            "road": {"fill": "#696969", "stroke": "#2F4F4F", "stroke-width": "0.5"},
            "field": {"fill": "#ADFF2F", "stroke": "#9ACD32", "stroke-width": "0.5"},
            "forest": {"fill": "#228B22", "stroke": "#006400", "stroke-width": "0.5"},
            "lake": {"fill": "#4682B4", "stroke": "#191970", "stroke-width": "0.5"},
            "river": {"fill": "#4682B4", "stroke": "#191970", "stroke-width": "0.5"},
            "hill": {"fill": "#8FBC8F", "stroke": "#556B2F", "stroke-width": "0.5"}
        }
        
        return style_map.get(category, {"fill": "#CCCCCC", "stroke": "#888888", "stroke-width": "0.5"})
    
    def create_enhanced_svg(self, image: Image.Image, quantized_image: np.ndarray, 
                           palette: List[Tuple[int, int, int]], segments_info: List[Dict],
                           width: int = 512, height: int = 512,
                           max_features: int = 100) -> str:
        """Create enhanced SVG using C++ logic"""
        
        # Calculate image center and max distance
        image_center = (quantized_image.shape[1] / 2.0, quantized_image.shape[0] / 2.0)
        max_dist_from_center = np.sqrt(image_center[0]**2 + image_center[1]**2)
        
        # Extract features for each color
        all_features = []
        
        for segment in segments_info:
            if segment['confidence'] < 0.1:  # Skip low-confidence segments
                continue
            
            color = segment['color']
            category = segment['category']
            
            # Extract contours for this color
            contours = self.extract_contours_for_color(quantized_image, color)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                importance = self.calculate_feature_importance(
                    contour, area, image_center, max_dist_from_center
                )
                
                # Get style for this category
                style = self.get_category_style_enhanced(category)
                
                all_features.append({
                    'contour': contour,
                    'color': color,
                    'category': category,
                    'area': area,
                    'importance': importance,
                    'style': style
                })
        
        # Sort by importance (descending)
        all_features.sort(key=lambda x: x['importance'], reverse=True)
        
        # Create SVG
        svg_parts = []
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {quantized_image.shape[1]} {quantized_image.shape[0]}">')
        
        # Add background
        img_array = np.array(image)
        avg_color = np.mean(img_array, axis=(0, 1)).astype(int)
        bg_color = self.compress_hex_color(avg_color[0], avg_color[1], avg_color[2])
        svg_parts.append(f'<rect width="{quantized_image.shape[1]}" height="{quantized_image.shape[0]}" fill="{bg_color}"/>')
        
        # Add features
        current_size = len(''.join(svg_parts))
        features_added = 0
        
        for feature in all_features:
            if features_added >= max_features:
                break
                
            if current_size > (self.MAX_SVG_SIZE_BYTES - self.SVG_SIZE_SAFETY_MARGIN):
                print(f"Warning: Approaching max SVG size, truncating output")
                break
            
            contour = feature['contour']
            style = feature['style']
            
            # Create polygon points
            points_str = ' '.join([f"{int(pt[0])},{int(pt[1])}" for pt in contour])
            
            # Create style string
            style_str = '; '.join([f"{k}: {v}" for k, v in style.items()])
            
            polygon_svg = f'<polygon points="{points_str}" style="{style_str}"/>'
            svg_parts.append(polygon_svg)
            
            current_size += len(polygon_svg)
            features_added += 1
        
        svg_parts.append('</svg>')
        
        print(f"Generated SVG with {features_added} features")
        return ''.join(svg_parts)
    
    def text_to_svg_enhanced(self, 
                           prompt: str, 
                           num_colors: int = 8,
                           width: int = 512, 
                           height: int = 512,
                           max_features: int = 100,
                           min_contour_area: float = 100,
                           simplification_epsilon: float = 0.02,
                           **generation_kwargs) -> Tuple[str, Image.Image]:
        """
        Enhanced text-to-SVG pipeline with C++ logic integration
        
        Args:
            prompt: Text description
            num_colors: Number of colors for quantization
            width: SVG width
            height: SVG height
            max_features: Maximum features to render
            min_contour_area: Minimum contour area to consider
            simplification_epsilon: Contour simplification factor
            **generation_kwargs: Additional arguments for image generation
        
        Returns:
            Tuple of (SVG string, generated image)
        """
        
        print(f"🎨 Generating image from prompt: '{prompt}'")
        # Step 1: Generate image
        image = self.generate_image(prompt, width=width, height=height, **generation_kwargs)
        
        print(f"🔍 Performing color quantization with {num_colors} colors")
        # Step 2: Enhanced color quantization
        img_array = np.array(image)
        quantized_image, palette = self.perform_color_quantization(img_array, num_colors)
        
        print(f"🏷️  Classifying segments using CLIP")
        # Step 3: Classify segments
        segments_info = self.classify_segments_enhanced(image, quantized_image, palette)
        
        print(f"📝 Found {len(segments_info)} segments")
        for seg in segments_info:
            print(f"   - {seg['category']}: {seg['confidence']:.2f} confidence")
        
        print(f"🎭 Creating enhanced SVG representation")
        # Step 4: Create enhanced SVG
        svg_content = self.create_enhanced_svg(
            image, quantized_image, palette, segments_info,
            width, height, max_features
        )
        
        print(f"✅ Enhanced SVG generation complete!")
        return svg_content, image
    
    def save_svg(self, svg_content: str, filename: str):
        """Save SVG to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"💾 SVG saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced text-to-SVG converter
    converter = EnhancedVisionLanguageTextToSVG()
    
    # Test prompts
    test_prompts = [
        "a beautiful landscape with mountains and trees",
        "a simple house with a garden",
        "a sunset over the ocean",
        "a forest with tall trees and flowers"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"Processing prompt {i+1}: {prompt}")
        print(f"{'='*60}")
        
        try:
            # Generate enhanced SVG
            svg_content, original_image = converter.text_to_svg_enhanced(
                prompt=prompt,
                num_colors=8,
                max_features=80,
                min_contour_area=150,
                simplification_epsilon=0.015,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            
            # Save results
            svg_filename = f"enhanced_svg_{i+1}.svg"
            img_filename = f"enhanced_original_{i+1}.png"
            
            converter.save_svg(svg_content, svg_filename)
            original_image.save(img_filename)
            
            print(f"🖼️  Original image saved to {img_filename}")
            
        except Exception as e:
            print(f"❌ Error processing prompt: {e}")
            continue
    
    print(f"\n🎉 All done! Check the enhanced SVG files.")