import base64
import requests
import json
import time
import os
import logging
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageEncoder:
    """Handles image encoding operations."""
    
    @staticmethod
    def encode_to_base64(image_path: str, resize: Optional[Tuple[int, int]] = None) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            resize: Optional tuple (width, height) to resize the image
            
        Returns:
            Base64 encoded string of the image
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        try:
            if resize:
                # Open the image, resize it, and convert to bytes
                with Image.open(image_path) as img:
                    img = img.resize(resize, Image.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format=img.format or "JPEG")
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                # Original behavior - no resize
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise

class PromptManager:
    """Manages different types of prompts for image analysis."""
    
    @staticmethod
    def get_structural_defect_prompt() -> str:
        """Returns the prompt for structural defect analysis."""
        return """
            This is an image from a sewer pipe inspection.

            It appears that there is a structural issue in the pipe. Please help classify the defect:

            Is the issue:
            - D – Deformation: A change in the shape or roundness of the pipe, such as flattening or bending  
            **OR**
            - X – Collapse: A major structural failure where the pipe wall has caved in or collapsed significantly

            Choose the most appropriate classification: D or X.

            📊 Also provide the **severity grade** of the defect using this scale:
            - 1 – Minor defect, no immediate action required
            - 2 – Noticeable, but not urgent
            - 3 – Moderate severity, monitoring recommended
            - 4 – Severe defect, action needed soon
            - 5 – Critical defect, immediate repair required

            ⚠️ IMPORTANT:
            - Respond ONLY with the defect code and severity number.
            - Do NOT explain or add extra text.
            - Example valid response: `D 3` or `X 4`

            Even if the image is unclear, make your best guess.
        """
    
    @staticmethod
    def get_deposit_prompt() -> str:
        """Returns the prompt for deposit analysis."""
        return """
Look at the sewer pipe and the surrounding ground.

There appears to be a deposit inside or around the pipe. Based on what you see, guess what kind of deposit it is and how severe it is.

🧪 Deposit Types:
- DEE – Encrustation (minerals/salts from groundwater, often around joints)
- DEF – Fouling (sewage sludge stuck to walls)
- DEG – Grease (greasy buildup above waterline)
- DEZ – Other attached deposits
- DES – Fine deposits (sand, silt, dust)
- DER – Coarse deposits (rocks, debris, waste)
- DEC – Hard/compacted (cement-like or hardened materials)
- DEX – Other settled material (unknown type)

🎯 What to do:
- Guess the deposit code based on the image of the pipe and surrounding ground.
- If unsure, make your best guess based on visual clues (color, texture, pattern).
- Respond with **only the code**, like this: `DEF` or `DER`
⚠️ IMPORTANT:
- Do NOT explain or add extra text.
- Example valid response: `DES` or `DER` or `DEF` or `DEE` or `DEG` or `DEZ` or `DEX`
- Just you have doubt or unsure or image is not clear or doesnt matter combination of two or more, Just Say the Code, No other text
- Just you have doubt or unsure or image is not clear or doesnt matter combination of two or more, Just Say the Code, No other text.
- Just you have doubt or unsure or image is not clear or doesnt matter combination of two or more, Just Say the Code, No other text.
- Just you have doubt or unsure or image is not clear or doesnt matter combination of two or more, Just Say the Code, No other text.
- return without `` 
        """

class LLavaAnalyzer:
    """Main class for analyzing sewer images using LLava model."""
    
    def __init__(self, model_name: str = "llava:13b", api_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize the LLava analyzer.
        
        Args:
            model_name: Name of the LLava model to use
            api_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.api_url = api_url
        self.prompt_manager = PromptManager()
        self.image_encoder = ImageEncoder()
        logger.info(f"Initialized LLavaAnalyzer with model: {model_name}")
    
    def analyze_image(self, image_path: str, prompt_type: str = "deposit") -> Tuple[str, float]:
        """
        Analyze a sewer image using the LLava model.
        
        Args:
            image_path: Path to the image file
            prompt_type: Type of analysis to perform ('structural' or 'deposit')
            
        Returns:
            Tuple containing (model_response, time_taken)
        """
        logger.info(f"Analyzing image: {image_path}")
        
        try:
            # Resize and encode image
            image_b64 = self.image_encoder.encode_to_base64(image_path)
            
            # Select prompt based on type
            if prompt_type == "structural":
                prompt = self.prompt_manager.get_structural_defect_prompt()
            else:  # Default to deposit
                prompt = self.prompt_manager.get_deposit_prompt()
            
            # Prepare payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64]
            }
            
            # Send request and process response
            return self._process_api_response(payload)
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            raise
    
    def _process_api_response(self, payload: Dict[str, Any]) -> Tuple[str, float]:
        """
        Process the API response from Ollama.
        
        Args:
            payload: Request payload for the API
            
        Returns:
            Tuple containing (model_response, time_taken)
        """
        start_time = time.time()
        response = requests.post(self.api_url, json=payload, stream=True)
        
        # Collect the full response
        full_response = ""
        
        # Stream output with timestamp
        
        
        for line in response.iter_lines():
            if line:
                current_time = time.time() - start_time
                decoded_line = line.decode("utf-8")
                #print(f"[{current_time:.2f}s] {decoded_line}")
                
                # Try to parse JSON
                try:
                    data = json.loads(decoded_line)
                    # If response contains content, add it to the full response
                    if "response" in data:
                        full_response += data["response"]
                except json.JSONDecodeError:
                    pass
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n--- COMPLETE RESPONSE ---")
        print(full_response)
        print(f"\nTotal response time: {total_time:.2f} seconds")
        
        return full_response, total_time

def main():
    """Main function to run the analysis on multiple images."""
    # List of image paths to analyze
    image_paths = [
        r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined\images\BX-201-_jpg.rf.ad157ebde09f33f66ee9e0467c4422cf.jpg",
        r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined\images\ZW-72-_jpg.rf.4ee31d88421e995155c124c8530dfaf7.jpg",
        r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined\images\BX-135-_jpg.rf.978da2722ca14ff45b805738db449d58.jpg",
        r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined\images\BX-1-_jpg.rf.08ebf7641d6be574e4388a7b660b5e08.jpg",
        r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined\images\image84_jpg.rf.3ddcd50e540a371dfb4177723c8c1dad.jpg"# Add more image paths as needed
    ]
    
    # Initialize analyzer
    analyzer = LLavaAnalyzer()
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_paths):
        logger.info(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
        try:
            response, time_taken = analyzer.analyze_image(image_path)
            results.append({
                "image_path": image_path,
                "response": response,
                "time_taken": time_taken
            })
            print(f"Image {idx+1}: Time taken: {time_taken:.2f} seconds")
            print(f"Image {idx+1}: Response: {response}")
            print("-" * 50)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    for idx, result in enumerate(results):
        print(f"Image {idx+1}: {os.path.basename(result['image_path'])}")
        print(f"Response: {result['response']}")
        print(f"Time: {result['time_taken']:.2f}s")
        print("-" * 30)

if __name__ == "__main__":
    main()
