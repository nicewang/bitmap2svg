import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO

class VisionLanguageTextToSVG:
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
        
        # Predefined semantic categories for SVG generation
        self.semantic_categories = [
            "sky", "cloud", "mountain", "tree", "grass", "water", "building", 
            "person", "animal", "flower", "rock", "sun", "moon", "star"
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
    
    def segment_image(self, image: Image.Image, n_segments: int = 8) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Segment image using color clustering and edge detection"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Color-based segmentation using K-means
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape back to image dimensions
        segmented = labels.reshape(img_array.shape[:2])
        
        # Extract individual segments
        segments = []
        for i in range(n_segments):
            mask = (segmented == i).astype(np.uint8) * 255
            segments.append(mask)
        
        return segmented, segments
    
    def classify_segments(self, image: Image.Image, segments: List[np.ndarray]) -> List[Dict]:
        """Classify each segment using CLIP"""
        segment_info = []
        
        for i, segment_mask in enumerate(segments):
            # Create masked image
            img_array = np.array(image)
            masked_img = img_array.copy()
            
            # Apply mask
            mask_3d = np.stack([segment_mask] * 3, axis=-1) / 255.0
            masked_img = (masked_img * mask_3d).astype(np.uint8)
            
            # Skip if segment is too small
            if np.sum(segment_mask) < 1000:  # Skip small segments
                continue
            
            # Convert to PIL
            pil_segment = Image.fromarray(masked_img)
            
            # Use CLIP to classify
            inputs = self.clip_processor(
                text=self.semantic_categories,
                images=pil_segment,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the same device as the model
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
                'segment_id': i,
                'category': predicted_category,
                'confidence': confidence,
                'mask': segment_mask,
                'area': np.sum(segment_mask > 0)
            })
        
        return segment_info
    
    def extract_shapes(self, segment_mask: np.ndarray) -> List[np.ndarray]:
        """Extract geometric shapes from segment mask"""
        # Find contours
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to simple coordinate array
            points = simplified.reshape(-1, 2)
            shapes.append(points)
        
        return shapes
    
    def generate_svg_path(self, points: np.ndarray, shape_type: str = "polygon") -> str:
        """Generate SVG path from points"""
        if len(points) < 3:
            return ""
        
        if shape_type == "smooth":
            # Generate smooth curves using quadratic Bezier
            path = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
            
            for i in range(1, len(points)):
                curr = points[i]
                if i == len(points) - 1:
                    path += f" L {curr[0]:.1f} {curr[1]:.1f}"
                else:
                    next_point = points[i + 1] if i + 1 < len(points) else points[0]
                    # Simple quadratic curve
                    control_x = (curr[0] + next_point[0]) / 2
                    control_y = (curr[1] + next_point[1]) / 2
                    path += f" Q {curr[0]:.1f} {curr[1]:.1f} {control_x:.1f} {control_y:.1f}"
            
            path += " Z"
        else:
            # Simple polygon
            path = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
            for point in points[1:]:
                path += f" L {point[0]:.1f} {point[1]:.1f}"
            path += " Z"
        
        return path
    
    def get_category_style(self, category: str) -> Dict[str, str]:
        """Get SVG style based on semantic category"""
        style_map = {
            "sky": {"fill": "#87CEEB", "stroke": "none"},
            "cloud": {"fill": "#FFFFFF", "stroke": "#CCCCCC", "stroke-width": "1"},
            "mountain": {"fill": "#8B7355", "stroke": "#654321", "stroke-width": "2"},
            "tree": {"fill": "#228B22", "stroke": "#006400", "stroke-width": "1"},
            "grass": {"fill": "#9ACD32", "stroke": "#6B8E23", "stroke-width": "1"},
            "water": {"fill": "#4682B4", "stroke": "#191970", "stroke-width": "1"},
            "building": {"fill": "#D2691E", "stroke": "#8B4513", "stroke-width": "2"},
            "person": {"fill": "#FDBCB4", "stroke": "#8B4513", "stroke-width": "2"},
            "animal": {"fill": "#DEB887", "stroke": "#8B7355", "stroke-width": "2"},
            "flower": {"fill": "#FF69B4", "stroke": "#DC143C", "stroke-width": "1"},
            "rock": {"fill": "#708090", "stroke": "#2F4F4F", "stroke-width": "1"},
            "sun": {"fill": "#FFD700", "stroke": "#FFA500", "stroke-width": "2"},
            "moon": {"fill": "#F0E68C", "stroke": "#BDB76B", "stroke-width": "1"},
            "star": {"fill": "#FFFF00", "stroke": "#FFD700", "stroke-width": "1"}
        }
        
        return style_map.get(category, {"fill": "#CCCCCC", "stroke": "#888888", "stroke-width": "1"})
    
    def create_svg(self, segments_info: List[Dict], width: int = 512, height: int = 512) -> str:
        """Create final SVG from classified segments"""
        svg_elements = []
        
        # Sort segments by area (largest first, so they appear in background)
        segments_info.sort(key=lambda x: x['area'], reverse=True)
        
        for segment in segments_info:
            if segment['confidence'] < 0.1:  # Skip low-confidence segments
                continue
            
            # Extract shapes from segment
            shapes = self.extract_shapes(segment['mask'])
            
            for shape_points in shapes:
                if len(shape_points) < 3:
                    continue
                
                # Determine shape type based on category
                shape_type = "smooth" if segment['category'] in ["cloud", "tree", "mountain"] else "polygon"
                
                # Generate SVG path
                path_data = self.generate_svg_path(shape_points, shape_type)
                
                if path_data:
                    # Get style for this category
                    style = self.get_category_style(segment['category'])
                    
                    # Create path element
                    style_str = "; ".join([f"{k}: {v}" for k, v in style.items()])
                    svg_elements.append(f'<path d="{path_data}" style="{style_str}"/>')
        
        # Create complete SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <title>Generated SVG from Text</title>
    {chr(10).join(svg_elements)}
</svg>'''
        
        return svg_content
    
    def text_to_svg(self, 
                   prompt: str, 
                   n_segments: int = 8,
                   width: int = 512, 
                   height: int = 512,
                   **generation_kwargs) -> Tuple[str, Image.Image]:
        """
        Complete text-to-SVG pipeline
        
        Args:
            prompt: Text description
            n_segments: Number of image segments to create
            width: SVG width
            height: SVG height
            **generation_kwargs: Additional arguments for image generation
        
        Returns:
            Tuple of (SVG string, generated image)
        """
        
        print(f"🎨 Generating image from prompt: '{prompt}'")
        # Step 1: Generate image
        image = self.generate_image(prompt, width=width, height=height, **generation_kwargs)
        
        print(f"🔍 Segmenting image into {n_segments} regions")
        # Step 2: Segment image
        segmented, segments = self.segment_image(image, n_segments)
        
        print(f"🏷️  Classifying segments using CLIP")
        # Step 3: Classify segments
        segments_info = self.classify_segments(image, segments)
        
        print(f"📝 Found {len(segments_info)} meaningful segments")
        for seg in segments_info:
            print(f"   - {seg['category']}: {seg['confidence']:.2f} confidence")
        
        print(f"🎭 Creating SVG representation")
        # Step 4: Create SVG
        svg_content = self.create_svg(segments_info, width, height)
        
        print(f"✅ SVG generation complete!")
        return svg_content, image
    
    def save_svg(self, svg_content: str, filename: str):
        """Save SVG to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"💾 SVG saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize the text-to-SVG converter
    converter = VisionLanguageTextToSVG()
    
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
            # Generate SVG
            svg_content, original_image = converter.text_to_svg(
                prompt=prompt,
                n_segments=6,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            
            # Save results
            svg_filename = f"generated_svg_{i+1}.svg"
            img_filename = f"original_image_{i+1}.png"
            
            converter.save_svg(svg_content, svg_filename)
            original_image.save(img_filename)
            
            print(f"🖼️  Original image saved to {img_filename}")
            
        except Exception as e:
            print(f"❌ Error processing prompt: {e}")
            continue
    
    print(f"\n🎉 All done! Check the generated SVG files.")
    