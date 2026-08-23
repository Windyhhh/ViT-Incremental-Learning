import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.lora_vit import LoRAViT
from src.models.classifier import PrototypeClassifier
from src.utils.config import Config
from src.utils.diagnostics import FeatureSpaceDiagnostics
from src.utils.contrastive_loss import contrastive_prototype_loss
from src.utils.distillation import compute_multi_layer_distillation
from src.utils.lora_merging import merge_loras_orthogonal_projection, merge_loras_sd_lora_inspired


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar100_subset(start_class, num_classes, transform):
    """获取CIFAR-100的子集"""
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    end_class = start_class + num_classes
    indices = []
    for idx, (_, label) in enumerate(dataset):
        if start_class <= label < end_class:
            indices.append(idx)
    return Subset(dataset, indices)


def get_full_test_set(transform):
    """获取完整的CIFAR-100测试集"""
    return datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)


def main():
    """主函数"""
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 设备设置
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # 数据变换
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # 初始化诊断工具
    diagnostics = FeatureSpaceDiagnostics()
    
    # 初始化原型分类器
    classifier = PrototypeClassifier(temperature=Config.CLASSIFICATION_TEMPERATURE)
    
    # 初始化合并的LoRA状态
    merged_lora_state = None
    
    # 任务循环
    for task_id in range(Config.NUM_TASKS):
        print(f"\n{'='*100}")
        print(f"Task {task_id + 1}/{Config.NUM_TASKS}")
        print(f"Classes: {task_id * Config.CLASSES_PER_TASK} - {(task_id + 1) * Config.CLASSES_PER_TASK - 1}")
        print(f"{'='*100}")
        
        # 获取当前任务的数据集
        start_class = task_id * Config.CLASSES_PER_TASK
        end_class = start_class + Config.CLASSES_PER_TASK
        current_classes = list(range(start_class, end_class))
        
        # 训练集
        train_dataset = get_cifar100_subset(start_class, Config.CLASSES_PER_TASK, transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=Config.NUM_WORKERS
        )
        
        # 测试集（用于诊断）
        test_dataset = get_full_test_set(transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS
        )
        
        # 初始化模型
        model = LoRAViT(
            model_name=Config.MODEL_NAME, 
            rank=Config.LORA_RANK, 
            alpha=Config.LORA_ALPHA
        ).to(device)
        
        # 加载之前合并的LoRA状态（如果有）
        if merged_lora_state is not None:
            model.set_lora_state_dict(merged_lora_state)
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LR, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.EPOCHS_PER_TASK
        )
        
        # 训练当前任务
        model.train()
        for epoch in range(Config.EPOCHS_PER_TASK):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS_PER_TASK}") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    
                    # 前向传播
                    features = model(images)
                    
                    # 提取原型（如果是第一个epoch）
                    if epoch == 0:
                        classifier.extract_prototypes(task_id, model, train_loader, current_classes, device)
                    
                    # 预测
                    predictions = classifier.predict(features, device)
                    
                    # 计算基础损失（原型对比损失）
                    loss = contrastive_prototype_loss(
                        features, 
                        labels, 
                        classifier.prototypes, 
                        temperature=Config.CONTRASTIVE_TEMPERATURE
                    )
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 更新统计信息
                    total_loss += loss.item() * images.size(0)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += images.size(0)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': total_loss / total_samples,
                        'acc': total_correct / total_samples
                    })
            
            # 更新学习率
            scheduler.step()
        
        # 保存当前任务的LoRA状态
        current_lora_state = model.get_lora_state_dict()
        
        # 合并LoRA状态
        if task_id == 0:
            merged_lora_state = current_lora_state
        else:
            if Config.MERGE_STRATEGY == "orthogonal_projection":
                merged_lora_state = merge_loras_orthogonal_projection(
                    merged_lora_state, 
                    current_lora_state, 
                    task_id + 1,  # task_id从1开始
                    projection_threshold=Config.OPCM_PROJECTION_THRESHOLD
                )
            elif Config.MERGE_STRATEGY == "sd_lora":
                merged_lora_state, _, _ = merge_loras_sd_lora_inspired(
                    merged_lora_state, 
                    current_lora_state, 
                    task_id + 1
                )
            else:
                # 默认使用正交投影
                merged_lora_state = merge_loras_orthogonal_projection(
                    merged_lora_state, 
                    current_lora_state, 
                    task_id + 1
                )
        
        # 创建用于诊断的模型副本
        native_model = LoRAViT(
            model_name=Config.MODEL_NAME, 
            rank=Config.LORA_RANK, 
            alpha=Config.LORA_ALPHA
        ).to(device)
        native_model.set_lora_state_dict(current_lora_state)
        
        merged_model = LoRAViT(
            model_name=Config.MODEL_NAME, 
            rank=Config.LORA_RANK, 
            alpha=Config.LORA_ALPHA
        ).to(device)
        merged_model.set_lora_state_dict(merged_lora_state)
        
        # 运行诊断
        task_diagnostics = diagnostics.analyze_feature_space_drift(
            task_id, 
            native_model, 
            merged_model, 
            test_loader, 
            current_classes, 
            classifier.prototypes, 
            device
        )
        diagnostics.add_diagnostics(task_diagnostics)
        
        # 保存检查点
        checkpoint_dir = Path("checkpoints") / f"task_{task_id+1}"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'lora_state_dict': current_lora_state,
            'merged_lora_state': merged_lora_state,
            'prototypes': classifier.prototypes,
            'epoch': Config.EPOCHS_PER_TASK,
            'task_id': task_id
        }, checkpoint_dir / "checkpoint.pth")
    
    # 打印累积诊断分析
    diagnostics.print_cumulative_analysis()
    
    print(f"\n{'='*100}")
    print("增量学习流程完成！")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
