import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import time
import argparse
import yaml
import torch.nn as nn
import ray
ray.init()

def expand_env_vars(config):
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)  # This will replace ${HOME} with the actual home directory
    else:
        return config
    
# dataset
class ImageNetDataset(Dataset):
    def __init__(self, img_dir, gt_file, transform=None, split=128):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        # read ground truth labels
        self.ground_truths = {}
        with open(gt_file, 'r') as f:
            for line in f.readlines()[:self.split]:
                img_name, label = line.strip().split()
                self.ground_truths[img_name] = int(label)
        
        # get image file names
        self.image_files = sorted(os.listdir(img_dir))[:self.split]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.ground_truths[img_name]
        # print(f'image_path: {img_path}')
        # print(f'label: {label}')
        return image, label
    
def evaluate_model_lat(model, dataloader, device, args, class_names):
    correct = 0
    total = 0
    model.to(device)
    print("Warmup ......")
    with torch.no_grad():
        image_count = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            image_count += images.size(0)
            if image_count >= args.warmup_iter:
                break
    dur = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            end_time = time.time()
            dur += (end_time - start_time)
            # log for debug use
            # for i in range(len(predicted)):
            #     predicted_class = class_names[predicted[i].item()]
            #     actual_class = class_names[labels[i].item()]
            #     print(f"Image {i+1}: Predicted class = {predicted_class}, Ground truth class = {actual_class}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_accuracy = correct / total
    total_throughput = total / dur
    print(f"latency: {dur / total:.4f} s/request ")
@ray.remote
def evaluate_model(model, dataloader, device, args, class_names):
    # num_gpus = torch.cuda.device_count()
    # print(f"Starting evaluation on GPU {device}, total gpus: {num_gpus}")
    model.to(device)
    correct = 0
    total = 0
    print("Warmup ......")
    with torch.no_grad():
        image_count = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            image_count += images.size(0)
            if image_count >= args.warmup_iter:
                break
    dur = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            end_time = time.time()
            dur += (end_time - start_time)
            # log for debug use
            # for i in range(len(predicted)):
            #     predicted_class = class_names[predicted[i].item()]
            #     actual_class = class_names[labels[i].item()]
            #     print(f"Image {i+1}: Predicted class = {predicted_class}, Ground truth class = {actual_class}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return dur, correct, total
    
def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument("--img_dir", type=str, default="/home/dataset/imageNet/val/images",
                        help="Directory containing the images.")
    parser.add_argument("--gt_file", type=str, default="/home/dataset/imageNet/val.txt",
                        help="File containing the ground truth labels.")
    parser.add_argument("--cls_file", type=str, default="/home/dataset/imageNet/imagenet_classes.txt",
                        help="path to imagenet_classes.txt.")
    parser.add_argument("--split", type=int, default=20480,
                        help="Split size for processing.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch-size.")
    parser.add_argument("--num_instance", type=int, default=1,
                        help="num_instance.")
    parser.add_argument("--gpus_per_node", type=int, default=2,
                        help="gpus_per_node.")
    parser.add_argument("--warmup_iter", type=int, default=128,
                        help="Split size for processing.")
    args = parser.parse_args()
    num_instance = args.num_instance
    gpus_per_node = args.gpus_per_node
    total_gpus = num_instance * gpus_per_node
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(script_dir, "../config.yaml")
    with open(config, "r") as f:
        config = expand_env_vars(yaml.safe_load(f))
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    # ladd classes
    with open(args.cls_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    

    # preprocess
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # path to dataset and ground truth
    img_dir = args.img_dir
    gt_file = args.gt_file
    model = models.resnet50(pretrained=True)
    model.eval()
    # latency
    imagenet_val_dataset = ImageNetDataset(img_dir, gt_file, transform=transform, split=int(args.split / 16))
    val_loader = DataLoader(imagenet_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    evaluate_model_lat(model, val_loader, "cuda:0", args, class_names)
    
    # create dataset and dataloader
    imagenet_val_dataset = ImageNetDataset(img_dir, gt_file, transform=transform, split=args.split)
    split_sizes = [len(imagenet_val_dataset) // total_gpus] * total_gpus
    split_sizes[-1] += len(imagenet_val_dataset) - sum(split_sizes) 
    datasets = random_split(imagenet_val_dataset, split_sizes)
    dataloaders = [
        DataLoader(subset, args.batch_size, shuffle=False, num_workers=4)
        for subset in datasets
    ]
    # choose to split or duplicate
    val_loader = DataLoader(imagenet_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'batch_size: {args.batch_size}')
    tasks = []
    for node in range(num_instance):
        for i in range(gpus_per_node):
            gpu_index = node * gpus_per_node + i
            # print(f'node: {node}, gpu: {i}')
            # tasks.append(evaluate_model.options(num_gpus=gpus_per_node).remote(model, 
            #                                    dataloaders[gpu_index], 
            #                                    f'cuda:{i}', args, class_names))
            tasks.append(evaluate_model.options(num_gpus=gpus_per_node).remote(model, 
                                               val_loader, 
                                               f'cuda:{i}', args, class_names))
    results = ray.get(tasks)
    max_duration = max(result[0] for result in results)
    total_correct = sum(result[1] for result in results)
    # total_samples = sum(result[2] for result in results)
    total_samples = sum(result[2] for result in results) 
    total_accuracy = total_correct / total_samples
    total_throughput = total_samples / max_duration
    print(f"distributed processed {total_samples} images in {max_duration:.4f} seconds")
    print(f"Total Accuracy: {total_accuracy * 100:.2f}%, Total images: {total_samples}")
    print(f"Total Throughput: {total_throughput:.2f} images/second")

main()
