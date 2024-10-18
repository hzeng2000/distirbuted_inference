import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import time
import argparse
import yaml
import ray

# 初始化 Ray
ray.init(address='auto')  # 自动连接到Ray集群

# dataset
class ImageNetDataset(Dataset):
    def __init__(self, img_dir, gt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 读取 ground truth 标签
        self.ground_truths = {}
        with open(gt_file, 'r') as f:
            for line in f.readlines():
                img_name, label = line.strip().split()
                self.ground_truths[img_name] = int(label)
        
        # 获取图片文件名
        self.image_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.ground_truths[img_name]
        return image, label

# 将 evaluate_model 函数转为 ray 远程任务
@ray.remote(num_gpus=1)
def evaluate_model(model, dataloader, device, class_names):
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    total_accuracy = correct / total
    total_throughput = total / (end_time - start_time)
    
    print(f"Device {device}: Total Accuracy: {total_accuracy * 100:.2f}%, Total images: {total}")
    print(f"Device {device}: Total Throughput: {total_throughput:.2f} images/second")
    
    return total_accuracy, total_throughput

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--img_dir", type=str, default="/home/dataset/imageNet/val/images",
                        help="Directory containing the images.")
    parser.add_argument("--gt_file", type=str, default="/home/dataset/imageNet/val.txt",
                        help="File containing the ground truth labels.")
    parser.add_argument("--cls_file", type=str, default="/home/dataset/imageNet/imagenet_classes.txt",
                        help="path to imagenet_classes.txt.")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="Ratio to split the dataset between nodes.")
    
    args = parser.parse_args()

    # 读取 class_names
    with open(args.cls_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_dir = args.img_dir
    gt_file = args.gt_file
    
    # 创建 dataset 并将其拆分为两部分
    dataset = ImageNetDataset(img_dir, gt_file, transform=transform)
    split = int(len(dataset) * args.split_ratio)
    dataset1, dataset2 = random_split(dataset, [split, len(dataset) - split])
    
    # 为每个节点创建 dataloader
    dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False, num_workers=4)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=4)

    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()

    # 在两个节点上运行任务
    task1 = evaluate_model.remote(model.to('cuda:0'), dataloader1, 'cuda:0', class_names)
    task2 = evaluate_model.remote(model.to('cuda:0'), dataloader2, 'cuda:0', class_names)
    
    # 获取任务结果
    results = ray.get([task1, task2])
    
    # 聚合结果
    total_accuracy = sum(result[0] for result in results) / 2
    total_throughput = sum(result[1] for result in results)
    
    print(f"Overall Accuracy: {total_accuracy * 100:.2f}%")
    print(f"Overall Throughput: {total_throughput:.2f} images/second")

if __name__ == "__main__":
    main()
