import torch
from datasets import load_dataset
from torchvision import transforms


class TinyImageNetProcessor:
    """Processor for Tiny ImageNet dataset to align it with ImageNet labels for model evaluation.

    It preprocesses images to ImageNet standards, maps labels between Tiny ImageNet and ImageNet,
    and evaluates model predictions with these mappings.
    """

    def __init__(self, imagenet_classes_path):
        """Initializes the processor with the path to ImageNet classes and loads the dataset.
        
        Args:
            imagenet_classes_path (str): Path to the file containing ImageNet class labels.
        """
        self.imagenet_classes_path = imagenet_classes_path
        self.dataset = load_dataset("zh-plus/tiny-imagenet")
        self.target_imagenet_to_tiny, self.target_tiny_to_imagenet = self._load_and_map_labels()

    def _load_and_map_labels(self):
        """Loads ImageNet labels from a file and creates mappings with the dataset labels.
        
        Returns:
            tuple: Two dictionaries for label mapping between ImageNet and Tiny ImageNet.
        """
        try:
            with open(self.imagenet_classes_path, "r") as file:
                lines = file.readlines()
        except IOError:
            raise FileNotFoundError("The ImageNet classes file was not found.")

        label_to_imagenet_idx = {line.split()[0]: idx for idx, line in enumerate(lines)}
        tiny_labels = {label: idx for idx, label in enumerate(self.dataset["train"].features["label"].names)}

        common_labels = set(label_to_imagenet_idx.keys()) & set(tiny_labels.keys())
        imagenet_to_tiny = {label_to_imagenet_idx[label]: tiny_labels[label] for label in common_labels}
        tiny_to_imagenet = {v: k for k, v in imagenet_to_tiny.items()}

        return imagenet_to_tiny, tiny_to_imagenet

    def get_image_label_tensors(self, num_samples=100):
        """Fetches and preprocesses a specified number of image samples.
        
        Args:
            num_samples (int): Number of samples to process.

        Returns:
            tuple: Tensors of images and their corresponding labels.
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        rgb_samples = []
        attempt = 0
        # Shuffle and select a subset of valid samples to find RGB images with valid labels
        while len(rgb_samples) < num_samples and attempt < 10 * num_samples:
            valid_samples = self.dataset["valid"].shuffle(seed=attempt).select(range(num_samples * 2))
            for sample in valid_samples:
                if sample["image"].mode == "RGB" and sample["label"] in self.target_imagenet_to_tiny.values():
                    rgb_samples.append((transform(sample["image"]), sample["label"]))
                if len(rgb_samples) == num_samples:
                        break
            attempt += 1

        images, labels = zip(*rgb_samples) if rgb_samples else ([], [])
        return torch.stack(images), torch.tensor(labels)

    def compute_accuracy(self, outputs, labels):
        """Computes the accuracy of model outputs compared to true labels.

        Args:
            outputs (torch.Tensor): Model outputs.
            labels (torch.Tensor): True labels.

        Returns:
            float: Accuracy metric.
        """
        imagenet_labels = torch.tensor([self.target_tiny_to_imagenet[label.item()] for label in labels])
        relevant_indices = [self.target_tiny_to_imagenet[label.item()] for label in labels]
        filtered_outputs = outputs[:, relevant_indices]

        predicted_labels = torch.tensor([relevant_indices[idx] for idx in filtered_outputs.argmax(dim=-1)])
        return (predicted_labels == imagenet_labels).float().mean().item()

    def compute_topk_accuracy(self, outputs, labels, k=5):
        """Computes top-k accuracy of the model outputs.

        Args:
            outputs (torch.Tensor): Model outputs.
            labels (torch.Tensor): True labels.
            k (int): Top k predictions to consider.

        Returns:
            float: Top-k accuracy metric.
        """
        imagenet_labels = torch.tensor([self.target_tiny_to_imagenet[label.item()] for label in labels])
        relevant_indices = [self.target_tiny_to_imagenet[label.item()] for label in labels]
        filtered_outputs = outputs[:, relevant_indices]

        topk_preds = filtered_outputs.topk(k, dim=-1).indices
        topk_labels = torch.tensor([[relevant_indices[pred] for pred in preds] for preds in topk_preds])

        correct = sum(imagenet_labels[i] in topk_labels[i] for i in range(len(imagenet_labels)))
        return correct / len(imagenet_labels)
