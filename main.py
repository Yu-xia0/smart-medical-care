import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据集路径
data_dir = r'D:\GameDownload\archive (2)'

# 设置设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小为32x32
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分训练集、验证集和测试集
train_size = int(0.4 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


class UltraSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltraSimpleCNN, self).__init__()

        # 只有一个卷积层
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层，输出为一个非常小的尺寸
        self.fc1 = nn.Linear(8 * 16 * 16, 16)  # 64x64 -> 16x16，减少神经元数量
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.9)  # 增加Dropout，丢弃80%的神经元
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        # 卷积层与池化层
        x = self.pool1(self.relu1(self.conv1(x)))

        # 展平数据
        x = x.view(x.size(0), -1)

        # 全连接层与Dropout
        x = self.relu2(self.fc1(x))
        x = self.dropout1(x)  # Dropout操作
        x = self.fc2(x)

        return x


# 获取类别数量
num_classes = len(dataset.classes)
model = UltraSimpleCNN(num_classes).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用较低的学习率

import torch
import matplotlib.pyplot as plt


def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, val_accuracies


# 训练模型
train_losses, val_losses, val_accuracies = train(model, train_loader, val_loader, criterion, optimizer, epochs=2)


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# 测试模型
test(model, test_loader)

# # 可视化训练和验证损失
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# 保存模型
torch.save(model.state_dict(), 'ultra_simple_cnn.pth')
print("Model saved successfully!")

import random


def infer_and_visualize(model, dataset, original_dataset, num_images=5):
    model.eval()
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)  # 随机选择索引
        image, label = dataset[idx]

        image_tensor = image.unsqueeze(0).to(device)  # 添加 batch 维度
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        predicted_label = original_dataset.classes[predicted.item()]
        true_label = original_dataset.classes[label]

        # 转换图像格式以供显示
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 0.5) + 0.5  # 反归一化到 [0, 1]

        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(f'预测: {predicted_label}\n真实: {true_label}', fontsize=10)

    plt.tight_layout()
    plt.show()


# 调用推理与可视化函数
infer_and_visualize(model, test_dataset, dataset, num_images=6)
