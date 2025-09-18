import time 
import torch  
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# —— 中文显示（可选）——
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# ====== DNN 模型（MLP）======
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),              # (N,1,28,28) -> (N, 784)
            torch.nn.Linear(28*28, 256), # 全连接层，28*28 是输入向量的长度，256 是该层输出向量的长度
            torch.nn.ReLU(), # 激活函数，对输入的每个元素执行 max(0, x) 操作，简单来说，它将负值变为 0，正值不变
            torch.nn.Dropout(0.2), # 适度正则，在训练时 随机丢弃（置为0）20%的神经元，防止模型在训练时过度依赖某些特征，增强模型的泛化能力
            torch.nn.Linear(256, 128), # 另一个全连接层，输入大小为 256（上一层的输出），输出大小为 128
            torch.nn.ReLU(), # 另一个 ReLU 激活层，确保网络中包含非线性因素，提升模型的拟合能力
            torch.nn.Linear(128, 10) # 最后一层是一个全连接层，输出大小为 10。这代表网络输出 10 个值，分别对应 MNIST 数据集中的 10 个类别（数字 0 到 9）。这些输出值是 logits，没有经过 Softmax，适合和 CrossEntropyLoss 搭配使用。
        )

    def forward(self, x):
        return self.model(x)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据预处理：张量化 + 简单归一化
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

BATCH_SIZE = 256
EPOCHS = 10

# 数据集与加载器
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
testData  = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

trainLoader = torch.utils.data.DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader  = torch.utils.data.DataLoader(testData,  batch_size=BATCH_SIZE)

# 初始化网络、损失与优化器
net = Net().to(device)
print(net)

lossF = torch.nn.CrossEntropyLoss()          # 期望 logits（未过 softmax）
optimizer = torch.optim.Adam(net.parameters())  # 可调 lr，例如 lr=1e-3

history = {'Test Loss': [], 'Test Accuracy': []}

# ====== 本次运行仅保留“最佳acc”的检查点（文件名含时间戳，不会覆盖其他运行）======
run_id = time.strftime("%Y%m%d-%H%M%S")
best_ckpt_path = f'best_dnn_run{run_id}.pth'
best_acc = -1.0  # 用 -1.0 确保第一轮一定会保存

# ====== 训练与评估 ======
for epoch in range(1, EPOCHS + 1):
    # —— 训练（累计整轮的平均指标）——
    net.train(True)
    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    processBar = tqdm(trainLoader, unit='step')
    for step, (imgs, labels) in enumerate(processBar):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)                   # 前向：DNN 接收展平后的图像
        loss = lossF(outputs, labels)         # CrossEntropy( logits, labels )
        preds = torch.argmax(outputs, dim=1)
        acc_batch = (preds == labels).float().mean()

        loss.backward()
        optimizer.step()

        # 累计整轮训练指标
        bs = labels.size(0)
        train_loss_sum += loss.item() * bs
        train_correct  += (preds == labels).sum().item()
        train_samples  += bs

        # 进度条里显示“当前batch”的训练指标
        processBar.set_description(f"[{epoch}/{EPOCHS}] Loss: {loss.item():.4f}, Acc: {acc_batch.item():.4f}")
    processBar.close()

    train_loss = train_loss_sum / train_samples
    train_acc  = train_correct  / train_samples

    # —— 测试（整份测试集的平均指标）——
    net.eval()
    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for t_imgs, t_labels in testLoader:
            t_imgs   = t_imgs.to(device)
            t_labels = t_labels.to(device)

            t_out  = net(t_imgs)
            t_loss = lossF(t_out, t_labels)

            bs = t_labels.size(0)
            total_loss   += t_loss.item() * bs
            total_correct += (torch.argmax(t_out, dim=1) == t_labels).sum().item()
            total_samples += bs

    test_loss = total_loss / total_samples
    test_acc  = total_correct / total_samples

    # —— 每轮结束后，明确打印“整轮”的 Train/Test 指标 —— 
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 记录历史
    history['Test Loss'].append(test_loss)
    history['Test Accuracy'].append(test_acc)

    # —— 若测试准确率更高，覆盖保存到“本次运行”的最佳文件（仅保存state_dict） ——
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state': net.state_dict(),         # 只存state_dict
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc,
            'test_loss': test_loss,
            'run_id': run_id,
        }, best_ckpt_path)
        print(f"已保存新的最佳模型（acc={best_acc:.4f}，epoch={epoch}）：{best_ckpt_path}")

# ====== 绘图（根据数据自动限制坐标范围）======
def nice_ylim(values, pad_frac=0.12, min_pad=1e-4):
    vals = np.asarray(values, dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if np.isclose(vmin, vmax):
        return vmin - min_pad, vmax + min_pad
    rng = vmax - vmin
    pad = max(rng * pad_frac, min_pad)
    return vmin - pad, vmax + pad

n_points = len(history['Test Loss'])
epochs_x = list(range(1, n_points + 1))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# 1) Test Loss
ax1.plot(epochs_x, history['Test Loss'], label='Test Loss')
ax1.grid(True)
ax1.set_ylabel('Loss')
ax1.set_title('DNN 在 MNIST 上的表现')
ax1.legend(loc='best')
ax1.set_xlim(1, max(1, n_points))
ax1.set_ylim(*nice_ylim(history['Test Loss']))

# 2) Test Accuracy
ax2.plot(epochs_x, history['Test Accuracy'], label='Test Accuracy', color='red')
ax2.grid(True)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='best')
ax2.set_xlim(1, max(1, n_points))
ax2.set_ylim(*nice_ylim(history['Test Accuracy']))

plt.tight_layout()
plt.show()

print(f"训练结束，最佳测试准确率: {best_acc:.4f}，文件: {best_ckpt_path}")
