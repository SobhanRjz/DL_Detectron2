import os
from PIL import Image, ImageEnhance

def change_contrast(image_path, output_path, contrast_factor=1.5):
	"""
	Changes the contrast of an image using PIL and saves it.
	
	Args:
		image_path (str): Path to the input image.
		output_path (str): Path to save the output image.
		contrast_factor (float): Contrast adjustment factor. Values > 1 increase contrast.
	"""
	# Create output directory if it doesn't exist
	output_dir = os.path.dirname(output_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Read image
	try:
		image = Image.open(image_path)
	except:
		raise ValueError(f"Could not read image at {image_path}")

	# Create enhancer and adjust contrast
	enhancer = ImageEnhance.Contrast(image)
	augmented = enhancer.enhance(contrast_factor)
	
	# Show original and augmented images
	image.show(title='Original Image')
	augmented.show(title='Augmented Image')
	
	# Save augmented image
	augmented.save(output_path)
	print(f"Processed image saved to {output_path}")

# Example usage
input_image = "C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/DataSets/images/train/01A-3101-001A-3100-01_frame20503_png_jpg.rf.73456e58a1a0bc34c99930bb7e1f29b1.jpg"
output_image = "C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/DataSets/images/train/01A-3101-001A-3100-01_frame20503_png_jpg.rf.73456e58a1a0bc34c99930bb7e1f29b1_contrast.jpg" 
change_contrast(input_image, output_image, contrast_factor=1.5)
