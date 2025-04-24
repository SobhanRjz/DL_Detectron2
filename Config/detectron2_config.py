# detectron_config.py
"""
Configuration module for Detectron2 specific settings.
"""
import os
from Config.basic_config import DEVICE, OUTPUT_PATH, DATA_PATH
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class DetectronConfig:

	# Global class variables
	N_CLUSTER = 5  # set base ROI Head layer number
	EPOCH = 20
	IMS_PER_BATCH = 2
	
	def __init__(self):
		numberOfImages = len(DatasetCatalog.get("my_dataset_train")) * (2) # 2 is the number of times the dataset is repeated change base on your augmentation
		batches_per_epoch = numberOfImages / self.IMS_PER_BATCH
		self.max_iter = int(self.EPOCH * batches_per_epoch)
		self.max_iter = int(50000)
		self.dataset_dicts = DatasetCatalog.get("my_dataset_train") # traindata set
		self.anchor_boxes = self._calculate_anchor_boxes()
		self.kmeans_ratios = self._calculate_anchor_ratios()
		self.IsMaskRCNN = True
		self.IsResume = False
		self.modelName = ""
		self.cfg = get_cfg()
		self._initialize_cfg()
		#self. _find_non_serializable()
		self._save_cfg()

	def _initialize_cfg(self):
		
		# Model settings
		# Unfreeze the configuration to add custom keys
		self.cfg.defrost()
		self._set_model_config()
		if self.IsResume:
			self._set_dataset_config()
			self._set_solver()
			self.cfg.freeze()
			return

		self._set_dataset_config()
		self._set_dataloader_config()
		self._set_input_config()
		self._set_test_config()
		self._set_fpn()
		self._set_proposal_generator()
		self._set_rpn()
		self._set_roi_heads()
		self._set_anchor_generator()
		self._set_solver()
		# Freeze the configuration after modifications
		self.cfg.freeze()

	def _set_model_config(self):
		"""Set model-specific configurations."""
		

		if self.IsResume:
			self.modelName = "mask_rcnn_X_101_32x8d_FPN_3x.yaml"
			model_config = os.path.join(OUTPUT_PATH, self.modelName) # yaml file

			self.cfg.SOLVER.PATIENCE = 2000
			self.cfg.SOLVER.WARMUP_STEPS = 1000
			self.cfg.DATASETS.TRAIN_REPEAT_FACTOR = [['my_dataset_train', 1.0]]
			self.cfg.merge_from_file(model_config)
			self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Ensure it matches the number of classes in your dataset

			self.cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_PATH, "model_final.pth") # checkpoint file
			self.cfg.MODEL.DEVICE = DEVICE
			self.cfg.OUTPUT_DIR = OUTPUT_PATH
			return

		if self.IsMaskRCNN:
			self.modelName = "mask_rcnn_X_101_32x8d_FPN_3x.yaml"
			model_config = model_zoo.get_config_file("COCO-InstanceSegmentation/{}".format(self.modelName))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/{}".format(self.modelName))
		else:
			self.modelName = "faster_rcnn_X_101_32x8d_FPN_3x.yaml"
			model_config = model_zoo.get_config_file("COCO-Detection/{}".format(self.modelName))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/{}".format(self.modelName))

		if self.IsResume:
			model_config = os.path.join(OUTPUT_PATH, self.modelName) # yaml file
			self.cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_PATH, "model_final.pth") # checkpoint file


		self.cfg.merge_from_file(model_config)
		self.cfg.MODEL.DEVICE = DEVICE
		self.cfg.OUTPUT_DIR = OUTPUT_PATH
		
		# Calculate dataset-specific mean and std values
		mean, std = self._calculate_mean_and_std(os.path.join(DATA_PATH, 'train'))
		self.cfg.MODEL.PIXEL_MEAN = [m * 255.0 for m in mean]  # Convert from [0,1] to [0,255] range
		self.cfg.MODEL.PIXEL_STD = [s * 255.0 for s in std]  # Convert from [0,1] to [0,255] range

	def _set_dataset_config(self):
		"""Set dataset-specific configurations."""
		self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
		self.cfg.DATASETS.TEST = ("my_dataset_valid",)
		
		self.cfg.DATASETS.TRAIN_REPEAT_FACTOR = [['my_dataset_train', 1.0]]
		#self.cfg.DATASETS.TRAIN_REPEAT_FACTOR = 1.0  # Define the custom key

	def _set_dataloader_config(self):
		"""Set DataLoader-specific configurations."""
		# Set number of workers based on CPU cores available
		self.cfg.DATALOADER.NUM_WORKERS = 4
		# Filter out images with no annotations to improve training efficiency
		self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
		# Use RepeatFactorTrainingSampler for better handling of class imbalance
		self.cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
		# Set repeat threshold to 0.001 for rare categories
		self.cfg.DATALOADER.REPEAT_THRESHOLD = 0.1
		# Enable square root sampling for balanced class distribution
		self.cfg.DATALOADER.REPEAT_SQRT = True
		# Set random seed for reproducibility
		self.cfg.SEED = 42

		self.cfg.DATALOADER.ASPECT_RATIO_GROUPING = True

	def _set_input_config(self):
		"""Set input-specific configurations."""
		self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
		self.cfg.INPUT.MIN_SIZE_TRAIN = (300, 400, 500)  # Multi-scale training
		self.cfg.INPUT.MAX_SIZE_TRAIN = 800  # Increased max size
		self.cfg.INPUT.MIN_SIZE_TEST = 400  # Increased min size for testing
		self.cfg.INPUT.MAX_SIZE_TEST = 800  # Increased max size for testing
		self.cfg.INPUT.FORMAT = "BGR"
		self.cfg.INPUT.RANDOM_FLIP = "horizontal"  # Add horizontal flip
		self.cfg.INPUT.CROP.ENABLED = True
		self.cfg.INPUT.CROP.TYPE = "relative_range"
		self.cfg.INPUT.CROP.SIZE = [0.8, 0.8]  # Random cropping


	def _set_test_config(self):
		"""Set test/evaluation-specific configurations."""
		self.cfg.TEST.EVAL_PERIOD = 500
		self.cfg.TEST.DETECTIONS_PER_IMAGE = 100  # Increase detection limit


	def _set_fpn(self):
		"""Sets the Feature Pyramid Network (FPN) options."""
		self.cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
		self.cfg.MODEL.FPN.OUT_CHANNELS = 256
		#self.cfg.MODEL.FPN.NORM = "GN"  # Group Normalization
		self.cfg.MODEL.FPN.FUSE_TYPE = "sum"  # "sum" or "avg"

	def _set_proposal_generator(self):
		"""Sets the proposal generator options."""
		self.cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"  # Current options: "RPN", "RRPN", "PrecomputedProposals"
		self.cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0  # Minimum height and width for proposals
		self.cfg.MODEL.KEYPOINT_ON = False  # Detect keypoints
		self.cfg.MODEL.LOAD_PROPOSALS = False  # Use precomputed proposals
		self.cfg.MODEL.MASK_ON = self.IsMaskRCNN  # Enable mask branch for instance segmentation


	def _set_rpn(self):
		"""Sets the Region Proposal Network (RPN) head options."""
		self.cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
		#self.cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # Default: ["res4"]
		self.cfg.MODEL.RPN.BOUNDARY_THRESH = -1
		self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
		self.cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
		self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256 # If the dataset has many objects and you want the RPN to learn from more samples / check memory
		self.cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
		self.cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
		self.cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
		self.cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
		self.cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0
		self.cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
		self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
		self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
		self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
		self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
		self.cfg.MODEL.RPN.NMS_THRESH = 0.7
		self.cfg.MODEL.RPN.CONV_DIMS = [-1]
		
	def _save_cfg(self):
		"""
		Saves the configuration to a YAML file in the output directory.
		"""
		os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists
		config_path = os.path.join(self.cfg.OUTPUT_DIR, "{}".format(self.modelName))
		with open(config_path, "w") as f:
			f.write(self.cfg.dump())  # Dump the configuration as YAML
		print(f"Configuration saved to {config_path}")
		
	def _find_non_serializable(self):
		import yaml
		for key, value in self.cfg.items():
			try:
				yaml.safe_dump({key: value})
			except yaml.representer.RepresenterError as e:
				print(f"Non-serializable field found: {key} -> {value} (type: {type(value)})")

		
	def _set_anchor_generator(self):
		# Ensure the number of sizes matches the number of IN_FEATURES
		num_features = len(self.cfg.MODEL.ROI_HEADS.IN_FEATURES)  # ["p2", "p3", "p4", "p5"]
		anchor_boxes = self.anchor_boxes
		
		# Adjust the number of sizes to match IN_FEATURES
		# if len(anchor_boxes) != num_features:
		# 	print(f"Adjusting anchor sizes from {len(anchor_boxes)} to match {num_features} IN_FEATURES.")
		# 	if len(anchor_boxes) > num_features:
		# 		anchor_boxes = anchor_boxes[:num_features]  # Truncate
		# 	else:
		# 		while len(anchor_boxes) < num_features:
		# 			anchor_boxes.append(anchor_boxes[-1])  # Repeat last size
			
		self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [(self.kmeans_ratios)]
		self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = ([[round(w[0])] for w in anchor_boxes])

	def _set_roi_heads(self):
		classes = MetadataCatalog.get("my_dataset_train").thing_classes
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
		self.cfg.MODEL.ROI_HEADS.CLASS_NAMES = classes
		self.cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
		self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
		self.cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
		self.cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
		self.cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = len(classes)

	def _set_solver(self):


		
		print(f"Total Iterations: ********************************* {self.max_iter} ***************************************")
		"""Sets the solver options."""
		#self.cfg.SOLVER.PATIENCE = 2000 # cfg not recognized
		self.cfg.SOLVER.IMS_PER_BATCH = self.IMS_PER_BATCH
		self.cfg.SOLVER.BASE_LR = 0.001  # Increased learning rate
		self.cfg.SOLVER.MAX_ITER = self.max_iter  # Set to 20,000
		self.cfg.SOLVER.STEPS = (int(self.max_iter * 0.6), int(self.max_iter * 0.8))  # Step decay
		self.cfg.SOLVER.WARMUP_ITERS = 1000  # Warm-up for 1,000 iterations
		self.cfg.SOLVER.WARMUP_STEPS = 1000
		self.cfg.SOLVER.BASE_LR_END = 0.0001  # Lower final learning rate
		self.cfg.SOLVER.MOMENTUM = 0.9
		self.cfg.SOLVER.NESTEROV = False
		self.cfg.SOLVER.RESCALE_INTERVAL = False
		self.cfg.SOLVER.BIAS_LR_FACTOR = 1.0
		self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
		self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
		self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
		self.cfg.SOLVER.WEIGHT_DECAY = 0.0005  # Add weight decay for regularization
		
	
	def _calculate_anchor_boxes(self, n_clusters=N_CLUSTER, _shouldPlot=False):
		import numpy as np
		from sklearn.cluster import KMeans

		widths = []
		heights = []
		for data in self.dataset_dicts:
			for bbox in data['annotations']:
				widths.append(bbox['bbox'][2])
				heights.append(bbox['bbox'][3])

		dimensions = np.stack((widths, heights), axis=1)

		# Initialize KMeans model and fit to data
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(dimensions)
		anchor_boxes = kmeans.cluster_centers_

			
		if _shouldPlot:
			import matplotlib
			import matplotlib.pyplot as plt
			matplotlib.use('TkAgg')  # Use Tkinter backend
			# Visualization
			plt.figure(figsize=(10, 6))
			plt.scatter(widths, heights, alpha=0.5, label='Data Points')
			plt.scatter(anchor_boxes[:, 0], anchor_boxes[:, 1], c='r', s=100, label='Anchor Boxes', alpha=0.5)
			plt.xlabel('Widths')
			plt.ylabel('Heights')
			plt.title('Scatter Plot of Widths and Heights')
			plt.legend()
			plt.grid(True)
			plt.show()

		print(anchor_boxes)
		return anchor_boxes

	def _calculate_anchor_ratios(self, n_clusters = N_CLUSTER):
		import numpy as np
		from sklearn.cluster import KMeans

		ratios = []
		for data in self.dataset_dicts:
			for bbox in data['annotations']:
				width, height = bbox['bbox'][2], bbox['bbox'][3]
				if width > 0 and height > 0:
					ratio = width / height
					ratios.append(ratio)

		# Convert ratios to a numpy array
		ratios = np.array(ratios).reshape(-1, 1)

		# Initialize KMeans model and fit to ratios
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(ratios)

		# Extract anchor ratios from cluster centers
		anchor_ratios = kmeans.cluster_centers_.flatten()
		Aspect = sorted([i[0] for i in kmeans.cluster_centers_.tolist()])
		
		print(f"Calculated Anchor Ratios: {anchor_ratios}")
		return Aspect
	def calculate_mean_std(dataset_dir):
		"""
		Calculate mean and std of RGB channels for images in a folder.

		Args:
			dataset_dir (str): Path to folder containing images.

		Returns:
			tuple: (mean, std) for RGB channels.
		"""
		import os
		from PIL import Image
		import torch
		from torchvision import transforms
		from tqdm import tqdm

		transform = transforms.Compose([
			transforms.Resize((640, 640)),
			transforms.ToTensor()
		])

		image_paths = [
			os.path.join(dataset_dir, f)
			for f in os.listdir(dataset_dir)
			if f.lower().endswith(('.jpg', '.jpeg', '.png'))
		]

		total_sum = torch.zeros(3)
		total_squared_sum = torch.zeros(3)
		total_pixels = 0

		for path in tqdm(image_paths, desc="Processing images"):
			image = Image.open(path).convert("RGB")
			tensor = transform(image)
			total_sum += tensor.sum(dim=[1, 2])
			total_squared_sum += (tensor ** 2).sum(dim=[1, 2])
			total_pixels += tensor.shape[1] * tensor.shape[2]

		mean = total_sum / total_pixels
		std = (total_squared_sum / total_pixels - mean ** 2).sqrt()

		return mean.tolist(), std.tolist()

	def get_cfg(self):
		return self.cfg