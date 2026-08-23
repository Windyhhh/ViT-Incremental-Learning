import torch


def procrustes_alignment(source_features, target_features):
    """
    Procrustes对齐: 找到最优旋转矩阵R,使得 source @ R 最接近 target
    参考论文: "Geometric Prototype Alignment for Class-Incremental Learning"

    Args:
        source_features: (N, D) 源特征
        target_features: (N, D) 目标特征

    Returns:
        R: (D, D) 最优旋转矩阵
    """
    try:
        # 中心化
        source_mean = source_features.mean(dim=0, keepdim=True)
        target_mean = target_features.mean(dim=0, keepdim=True)

        source_centered = source_features - source_mean
        target_centered = target_features - target_mean

        # 计算协方差矩阵
        H = source_centered.T @ target_centered

        # 添加正则化以改善条件数
        H = H + 1e-6 * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        # SVD分解 (使用full_matrices=False以提高稳定性)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)

        # 最优旋转矩阵
        R = Vh.T @ U.T

        # 确保是旋转矩阵 (det(R) = 1)
        if torch.det(R) < 0:
            Vh_corrected = Vh.clone()
            Vh_corrected[-1, :] *= -1
            R = Vh_corrected.T @ U.T

        return R

    except Exception as e:
        # 如果SVD失败,返回单位矩阵 (无对齐)
        print(f"[WARNING] Procrustes alignment failed: {e}")
        return torch.eye(source_features.shape[1], device=source_features.device, dtype=source_features.dtype)


def compute_procrustes_alignment_loss(current_features, old_features, alignment_matrix=None):
    """
    Procrustes对齐损失: 鼓励当前特征与对齐后的旧特征相近

    Args:
        current_features: 当前模型特征 [batch, hidden_dim]
        old_features: 旧模型特征 [batch, hidden_dim]
        alignment_matrix: 预计算的对齐矩阵 (如果为None,则计算)

    Returns:
        对齐损失
    """
    try:
        if alignment_matrix is None:
            # 计算对齐矩阵
            alignment_matrix = procrustes_alignment(old_features, current_features)

        # 对齐旧特征
        aligned_old_features = old_features @ alignment_matrix

        # 计算对齐损失 (MSE)
        loss = F.mse_loss(current_features, aligned_old_features)

        return loss
    except Exception as e:
        # 如果计算失败,返回零损失
        print(f"[WARNING] Procrustes alignment loss failed: {e}")
        return torch.tensor(0.0, device=current_features.device, requires_grad=True)


def compute_procrustes_alignment_loss(current_features, old_features,
                                     alignment_matrices=None, temperature=4.0):
    """
    计算Procrustes对齐总损失 (Phase 2核心)

    Args:
        current_features: 当前特征 [batch, hidden_dim]
        old_features: 旧特征 [batch, hidden_dim]
        alignment_matrices: 预计算的对齐矩阵字典
        temperature: 温度参数

    Returns:
        对齐损失
    """
    import torch.nn.functional as F
    if old_features is None:
        return torch.tensor(0.0, device=current_features.device)

    # 方法1: 直接MSE损失
    alignment_loss = compute_procrustes_alignment_loss(current_features, old_features)

    # 方法2: 相似度匹配 (可选)
    current_norm = F.normalize(current_features, p=2, dim=1)
    old_norm = F.normalize(old_features, p=2, dim=1)

    # 计算相似度矩阵
    similarity = current_norm @ old_norm.T / temperature

    # 目标: 对角线上的相似度应该最高
    batch_size = current_features.shape[0]
    target = torch.arange(batch_size, device=current_features.device)

    # 相似度损失
    similarity_loss = F.cross_entropy(similarity, target)

    # 组合损失 (50% MSE + 50% 相似度)
    total_loss = 0.5 * alignment_loss + 0.5 * similarity_loss

    return total_loss
