import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image
import os
import time
import argparse
import yaml
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

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
    
def evaluate_model(model, dataloader, device, args, class_names):
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
    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # log for debug use
            # for i in range(len(predicted)):
            #     predicted_class = class_names[predicted[i].item()]
            #     actual_class = class_names[labels[i].item()]
            #     print(f"Image {i+1}: Predicted class = {predicted_class}, Ground truth class = {actual_class}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    total_correct = torch.tensor(correct).to(device)
    total_samples = torch.tensor(total).to(device)
    total_time = torch.tensor(end_time - start_time).to(device)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_time, op=dist.ReduceOp.MAX)

    if device == 0:
        total_accuracy = total_correct.item() / total_samples.item()
        total_throughput = total_samples.item() / total_time.item()
        print(f"Total Accuracy: {total_accuracy * 100:.2f}%, Total images: {total_samples.item()}")
        print(f"Total Throughput: {total_throughput:.2f} images/second")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def main(rank, world_size):
    device = rank
    setup(rank=rank, world_size=world_size)
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument("--img_dir", type=str, default="/home/dataset/imageNet/val/images",
                        help="Directory containing the images.")
    parser.add_argument("--gt_file", type=str, default="/home/dataset/imageNet/val.txt",
                        help="File containing the ground truth labels.")
    parser.add_argument("--cls_file", type=str, default="/home/dataset/imageNet/imagenet_classes.txt",
                        help="path to imagenet_classes.txt.")
    parser.add_argument("--split", type=int, default=128,
                        help="Split size for processing.")
    parser.add_argument("--warmup_iter", type=int, default=128,
                        help="Split size for processing.")

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(script_dir, "../config.yaml")
    with open(config, "r") as f:
        config = yaml.safe_load(f)
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

    # create dataset and dataloader
    imagenet_val_dataset = ImageNetDataset(img_dir, gt_file, transform=transform, split=args.split)
    sampler = torch.utils.data.distributed.DistributedSampler(imagenet_val_dataset, num_replicas=torch.cuda.device_count(), rank=rank)
    val_loader = DataLoader(imagenet_val_dataset, batch_size=32, shuffle=False, num_workers=4, sampler=sampler)
    model = models.resnet50(pretrained=True).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.eval()

    evaluate_model(model, val_loader, device, args, class_names)
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting ddp in {world_size} devices")
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)