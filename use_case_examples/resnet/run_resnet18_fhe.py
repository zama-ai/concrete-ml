import argparse
import torch
from resnet import ResNet18_Weights, resnet18_custom
from concrete.ml.torch.compile import compile_torch_model
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Run ResNet18 model with FHE execution.")
parser.add_argument('--run_fhe', action='store_true', help="Run the actual FHE execution.")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

# Load the ResNet18 model with pretrained weights
resnet18 = resnet18_custom(weights=ResNet18_Weights.IMAGENET1K_V1)

# Use ImageNet classes file to map class names to indices
imagenet_classes_path = BASE_DIR / "imagenet_classes.txt"
with open(imagenet_classes_path, "r") as f:
    class_to_index = {cls: idx for idx, cls in enumerate([line.strip() for line in f.readlines()])}

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TODO: have a more automated way to grab N images from the net.
# Download an example image from the web
image_urls = [
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01614925_bald_eagle.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01697457_African_crocodile.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01592084_chickadee.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01601694_water_ouzel.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01739381_vine_snake.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01806567_quail.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01917289_brain_coral.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02077923_sea_lion.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02051845_pelican.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02110185_Siberian_husky.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02165456_ladybug.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02325366_wood_rabbit.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02391049_zebra.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02481823_chimpanzee.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02510455_giant_panda.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02643566_lionfish.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02787622_banjo.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02817516_bearskin.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02871525_bookshop.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02930766_cab.JPEG",
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02974003_car_wheel.JPEG",
]

images, labels = [], []
for image_url in image_urls:
    class_name = '_'.join(image_url.split('/')[-1].split('.')[0].split('_')[1:]).replace('_', ' ')
    if class_name in class_to_index:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        images.append(transform(img))
        labels.append(class_to_index[class_name])

# Stack images to create a mini batch
images = torch.stack(images)
labels = torch.tensor(labels)

# Function to compute accuracy
def compute_accuracy(predicted, labels):
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

# Function to compute top-k accuracy
def compute_topk_accuracy(outputs, labels, topk=5):
    _, topk_predicted = torch.topk(outputs, topk, dim=1)
    correct_topk = sum([labels[i] in topk_predicted[i] for i in range(len(labels))])
    total = labels.size(0)
    return 100 * correct_topk / total

# Forward pass through the model to get the predictions
with torch.no_grad():
    outputs = resnet18(images)
    _, predicted = torch.max(outputs, 1)

# Compute and print accuracy
accuracy = compute_accuracy(predicted, labels)
print(f"Accuracy of the ResNet18 model on the images: {accuracy:.4f}%")

topk_accuracy = compute_topk_accuracy(outputs, labels, topk=5)
print(f"Top-5 Accuracy of the ResNet18 model on the images: {topk_accuracy:.4f}%")

# Compile the model
print("Compiling the model...")
q_module = compile_torch_model(
    resnet18,
    torch_inputset=images,
    n_bits={"model_inputs": 8, "op_inputs": 7, "op_weights": 6, "model_outputs": 8},
    rounding_threshold_bits={"n_bits": 7, "method":"APPROXIMATE"},
    p_error=0.005
)
print("Model compiled successfully.")

# Forward pass with FHE disabled
with torch.no_grad():
    outputs_disable = q_module.forward(images.detach().numpy(), fhe="disable")
    _, predicted_disable = torch.max(torch.from_numpy(outputs_disable), 1)

# Compute accuracy
fhe_accuracy_vs_fp32 = (predicted_disable == predicted).float().mean().item()
print(f"Quantized Model Fidelity with FP32: {fhe_accuracy_vs_fp32:.4f}%")

# Compute and print accuracy for quantized model
accuracy = compute_accuracy(predicted_disable, labels)
print(f"Quantized Model Accuracy of the FHEResNet18 on the images: {accuracy:.4f}%")
topk_accuracy = compute_topk_accuracy(torch.from_numpy(outputs_disable), labels, topk=5)
print(f"Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: {topk_accuracy:.4f}%")

# Forward pass with FHE simulation
with torch.no_grad():
    outputs_simulate = q_module.forward(images.detach().numpy(), fhe="simulate")
    _, predicted_simulate = torch.max(torch.from_numpy(outputs_simulate), 1)

# Compute and print accuracy for FHE simulation
accuracy = compute_accuracy(predicted_simulate, labels)
print(f"FHE Simulation Accuracy of the FHEResNet18 on the images: {accuracy:.4f}%")
topk_accuracy = compute_topk_accuracy(torch.from_numpy(outputs_simulate), labels, topk=5)
print(f"FHE Simulation Top-5 Accuracy of the FHEResNet18 on the images: {topk_accuracy:.4f}%")

if args.run_fhe:
    # Run FHE execution and measure time on a single image
    q_module.fhe_circuit.keygen()
    single_image = images[0:1].detach().numpy()
    
    start = time.time()
    fhe_output = q_module.forward(single_image, fhe="simulate")
    end = time.time()
    print(f"Time taken for one FHE execution: {end - start:.4f} seconds")
    print(f"FHE execution output: {fhe_output}")

    # Run FHE simulation on the same single image
    fhe_sim_output = q_module.forward(single_image, fhe="simulate")
    print(f"FHE simulation output: {fhe_sim_output}")
    print(f"Actual label: {labels[0].item()}")

else:
    print("FHE execution was not run. Use --run_fhe to enable it.")