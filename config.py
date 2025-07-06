import torch
import os


class Config:
    def __init__(self):
        # 随机种子
        self.seed = 42

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 数据配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, "Raw_data")
        self.output_dir = os.path.join(current_dir, "Augmented_data")
        self.test_ratio = 0.2

        # 扩散模型配置
        self.diffusion_input_dim = 1280  # ESM嵌入维度
        self.diffusion_T = 500  # 时间步数
        self.diffusion_epochs = 100  # 训练轮数
        self.diffusion_batch_size = 64  # 批大小
        self.diffusion_lr = 1e-4  # 学习率

        # 增强配置
        self.target_ratio = 0.4  # 目标正样本比例
        self.min_samples_per_protein = 30  # 最少生成样本数
        self.knn_k = 5  # KNN图的邻居数
        self.oversample_ratio = 2.0  # 过采样比例
        self.max_nodes_per_graph = 1000  # 每张图的最大节点数

        # GNN模型配置
        self.gnn_hidden_dim = 256  # 减小隐藏层维度
        self.gnn_epochs = 100  # 训练轮数
        self.gnn_lr = 5e-4  # 学习率
        self.gnn_dropout = 0.3
        self.gnn_patience = 15  # 早停耐心值

        # 保存选项
        self.save_diffusion_model = True
        self.save_augmented_data = True

        # 打印路径信息
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")