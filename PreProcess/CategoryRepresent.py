import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from Config.basic_config import DATA_PATH
import os
class COCOAnalyzer:
    def __init__(self):
        annotation_file = os.path.join(DATA_PATH, "Final.json")
        self.annotation_file = annotation_file
        self.coco_data = None
        self.categories = None
        self.annotations = None
        self.bbox_count = defaultdict(int)
        self.category_id_to_name = {}
        self.category_data = None
        self.category_names = None
        self.bbox_counts = None
        self.clusters = None
        
    def load_data(self):
        """Load and validate COCO annotation file"""
        try:
            with open(self.annotation_file, "r") as f:
                self.coco_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find annotation file at {self.annotation_file}")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.annotation_file}")
            exit(1)
            
        self.categories = self.coco_data.get("categories", [])
        self.annotations = self.coco_data.get("annotations", [])
        
        if not self.categories or not self.annotations:
            print("Error: No categories or annotations found in the JSON file")
            exit(1)
            
    def process_data(self):
        """Process annotations and prepare data for visualization"""
        # Count bounding boxes per category
        for ann in self.annotations:
            category_id = ann.get("category_id")
            if category_id is not None:
                self.bbox_count[category_id] += 1
                
        # Map category IDs to names
        self.category_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        
        # Prepare and sort data
        self.category_data = [(self.category_id_to_name[cat_id], count) 
                             for cat_id, count in self.bbox_count.items()]
        self.category_data.sort(key=lambda x: x[1], reverse=True)
        self.category_names, self.bbox_counts = zip(*self.category_data)
        
    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering on bbox counts"""
        X = np.array(self.bbox_counts).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(X)
        return self.clusters
        
    def visualize_distribution(self, n_clusters=4):
        """Create visualization of the distribution"""
        colors = [f'#{np.random.randint(0, 0xFFFFFF):06x}' for _ in range(n_clusters)]
        
        bar_colors = [colors[cluster] for cluster in self.clusters]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.category_names, self.bbox_counts, color=bar_colors)
        plt.xlabel("Categories")
        plt.ylabel("Number of Bounding Boxes")
        plt.title("Bounding Box Distribution per Category (Sorted by Count)")
        plt.xticks(rotation=45, ha="right")
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i],
                                       label=f'Cluster {i+1}') for i in range(n_clusters)]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig('category_distribution.png')
        plt.waitforbuttonpress()
        plt.close()
        
    def print_analysis(self, n_clusters=4):
        """Print cluster analysis and category mapping"""
        print("\nCluster Analysis:")
        for i in range(n_clusters):
            cluster_categories = [name for name, cluster in zip(self.category_names, self.clusters) 
                                if cluster == i]
            cluster_counts = [count for count, cluster in zip(self.bbox_counts, self.clusters) 
                            if cluster == i]
            print(f"\nCluster {i+1}:")
            print(f"Average count: {np.mean(cluster_counts):.2f}")
            print(f"Categories: {', '.join(cluster_categories)}")
            
        print("\nCategory ID to Name mapping:")
        print(json.dumps(self.category_id_to_name, indent=2))

def main():
    
    analyzer = COCOAnalyzer()
    analyzer.load_data()
    analyzer.process_data()
    analyzer.perform_clustering()
    analyzer.visualize_distribution()
    analyzer.print_analysis()

if __name__ == "__main__":
    main()