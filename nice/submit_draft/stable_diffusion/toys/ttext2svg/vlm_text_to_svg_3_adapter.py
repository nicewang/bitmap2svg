import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict
from transformers import CLIPProcessor, CLIPModel

import kagglehub

class EnhancedVisionLanguageTextToSVG:
    def __init__(self):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        self.model_id = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

        # Initialize Stable Diffusion pipeline
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,  # Use half precision
            safety_checker=None         # Disable safety checker for speed
        )
        self.pipe = self.pipe.to(self.device)
        
        # Initialize CLIP for semantic understanding
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model_path = kagglehub.model_download('nicecaliforniaw/openai-clip-vit-large-patch14/Transformers/default/1')
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_path)
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
        
    def generate_image(self, prompt: str, negative_prompt: str, **kwargs) -> Image.Image:
        """Generate image from text prompt using Stable Diffusion"""
        default_kwargs = {
            "num_inference_steps": 27,
            "guidance_scale": 20,
            "height": 512,
            "width": 512
        }
        default_kwargs.update(kwargs)
        
        with torch.no_grad():
            result = self.pipe(prompt=prompt, negative_prompt=negative_prompt, **default_kwargs)
        
        torch.cuda.empty_cache()
        
        return result.images[0]
    
    def compress_hex_color(self, r: int, g: int, b: int) -> str:
        """Compress hex color representation (from C++ logic)"""
        # Check if we can use 3-digit hex (like #RGB instead of #RRGGBB)
        if (r % 17 == 0 and g % 17 == 0 and b % 17 == 0 and 
            r // 17 < 16 and g // 17 < 16 and b // 17 < 16):
            return f"#{r//17:x}{g//17:x}{b//17:x}"
        else:
            return f"#{r:02x}{g:02x}{b:02x}"
    
    def perform_color_quantization(self, image: np.ndarray, num_colors_hint: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Enhanced color quantization using K-means with full adaptive logic from C++
        
        Args:
            image: Input RGB image
            num_colors_hint: Color count hint (0 for automatic, >0 for manual)
        
        Returns:
            Tuple of (quantized_image, palette)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with 3 channels")
        
        # Adaptive color count based on image size (exactly matching C++ logic)
        k = num_colors_hint
        if k <= 0:
            pixel_count = image.shape[0] * image.shape[1]
            if pixel_count == 0:
                k = 1
            elif pixel_count < 16384:  # < 128x128
                k = 6
            elif pixel_count < 65536:  # < 256x256
                k = 8
            elif pixel_count < 262144:  # < 512x512
                k = 10
            else:
                k = 12
        
        # Ensure k is within reasonable range [1, 16] for automatic selection
        # But allow user override if num_colors_hint was > 0 initially
        if num_colors_hint <= 0:
            k = max(1, min(k, 16))
        else:
            k = max(1, k)  # Only ensure minimum of 1 for manual selection
        
        # Reshape image for K-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        if pixels.shape[0] == 0:
            print("Error: No samples to process for k-means (image might be empty)")
            return image.astype(np.uint8), [(128, 128, 128)]  # Fallback
        
        # Handle case where we have fewer pixels than requested clusters
        if pixels.shape[0] < k:
            k = pixels.shape[0]
            if k == 0:
                return image.astype(np.uint8), [(128, 128, 128)]  # Fallback
        
        # Check for unique colors to avoid unnecessary clustering
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) <= k:
            # If we have fewer unique colors than requested, use them directly
            k = len(unique_colors)
            centers = unique_colors.astype(np.float32)
            
            # Create labels by finding closest center for each pixel
            labels = np.zeros(pixels.shape[0], dtype=np.int32)
            for i, pixel in enumerate(pixels):
                distances = np.sum((centers - pixel) ** 2, axis=1)
                labels[i] = np.argmin(distances)
        else:
            # Perform K-means clustering with adaptive parameters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            attempts = 5 if k <= 8 else 7
            
            try:
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
                )
            except cv2.error as e:
                print(f"K-means failed: {e}. Using fallback method.")
                # Fallback to simple uniform sampling
                indices = np.linspace(0, len(unique_colors) - 1, k, dtype=int)
                centers = unique_colors[indices].astype(np.float32)
                labels = np.zeros(pixels.shape[0], dtype=np.int32)
                for i, pixel in enumerate(pixels):
                    distances = np.sum((centers - pixel) ** 2, axis=1)
                    labels[i] = np.argmin(distances)
        
        # Handle empty centers (fallback logic from C++)
        if centers.shape[0] == 0:
            print("Warning: k-means returned 0 centers. Using average color fallback.")
            avg_color = np.mean(image, axis=(0, 1))
            centers = avg_color.reshape(1, -1).astype(np.float32)
            labels = np.zeros(pixels.shape[0], dtype=np.int32)
            k = 1
        
        # Create quantized image
        quantized_pixels = centers[labels.flatten()]
        quantized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)
        
        # Extract palette with proper bounds checking
        palette = []
        for i in range(centers.shape[0]):
            r = int(np.clip(centers[i, 0], 0, 255))
            g = int(np.clip(centers[i, 1], 0, 255))
            b = int(np.clip(centers[i, 2], 0, 255))
            palette.append((r, g, b))
        
        print(f"Color quantization completed: {len(palette)} colors from {k} requested")
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
        
        return ''.join(svg_parts)
    
    def text_to_svg_enhanced(self, 
                           prompt: str,
                           negative_prompt: str, 
                           num_colors: int = 0,  # 0 for automatic, >0 for manual
                           width: int = 512, 
                           height: int = 512,
                           max_features: int = 100,
                           min_contour_area: float = 100,
                           simplification_epsilon: float = 0.02,
                           **generation_kwargs) -> Tuple[str, Image.Image]:
        """
        Enhanced text-to-SVG pipeline with full adaptive color quantization
        
        Args:
            prompt: Text description
            num_colors: Number of colors (0 for automatic adaptation, >0 for manual)
            width: SVG width
            height: SVG height
            max_features: Maximum features to render
            min_contour_area: Minimum contour area to consider
            simplification_epsilon: Contour simplification factor
            **generation_kwargs: Additional arguments for image generation
        
        Returns:
            Tuple of (SVG string, generated image)
        """
        
        # Step 1: Generate image
        image = self.generate_image(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, **generation_kwargs)
        
        # Step 2: Enhanced color quantization with full adaptive logic
        img_array = np.array(image)
        quantized_image, palette = self.perform_color_quantization(img_array, num_colors)
        
        # Step 3: Classify segments
        segments_info = self.classify_segments_enhanced(image, quantized_image, palette)
        
        print(f"🎭 Creating enhanced SVG representation")
        # Step 4: Create enhanced SVG
        svg_content = self.create_enhanced_svg(
            image, quantized_image, palette, segments_info,
            width, height, max_features
        )
        
        return svg_content, image

# Example usage
if __name__ == "__main__":
    converter = EnhancedVisionLanguageTextToSVG()

    svg_content, image = converter.text_to_svg_enhanced(
        prompt="a lighthouse overlooking the ocean",
        negative_prompt='',
        num_colors=0,           
        max_features=100,        
        min_contour_area=1.0,   
        simplification_epsilon=0.009,  
        num_inference_steps=27,
        guidance_scale=20
    )
