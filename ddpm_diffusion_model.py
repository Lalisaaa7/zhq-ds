import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.embed = nn.Embedding(T, dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.linear(self.embed(t))


class DiffusionPredictor(nn.Module):
    def __init__(self, input_dim, T, protein_dim=1280):
        super().__init__()
        # 时间嵌入
        self.time_embed = TimeEmbedding(T, 256)

        # 蛋白质结构条件编码器
        self.protein_cond = nn.Sequential(
            nn.Linear(protein_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        # 增强核心网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + 256 + 256, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x_t, t, protein_context):
        # 时间嵌入
        te = self.time_embed(t)

        # 蛋白质条件
        pc = self.protein_cond(protein_context)

        # 融合输入
        combined = torch.cat([x_t, te, pc], dim=-1)
        return self.net(combined)


class DiffusionProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # 在指定设备上创建张量
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def diffuse(self, x_0, t):
        """前向扩散过程"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def sample_timesteps(self, n):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (n,), device=self.device)


class EnhancedDiffusionModel(nn.Module):
    def __init__(self, input_dim, T=500, protein_dim=1280, device='cpu'):
        super().__init__()
        self.T = T
        self.device = device
        self.input_dim = input_dim
        self.protein_dim = protein_dim
        self.has_positive_samples = False

        # 扩散模型组件
        self.predictor = DiffusionPredictor(input_dim, T, protein_dim).to(device)
        self.diffusion = DiffusionProcess(num_timesteps=T, device=device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train_on_positive_samples(self, all_data, epochs=100, batch_size=64):
        """训练扩散模型"""
        positive_vectors = []
        protein_contexts = []

        # 收集正样本及其蛋白质上下文
        for data in all_data:
            x = data.x.to(self.device)
            y = data.y.to(self.device)
            pos_mask = (y == 1)

            if pos_mask.sum() > 0:
                self.has_positive_samples = True
                pos_feats = x[pos_mask]
                protein_ctx = data.protein_context.to(self.device)

                positive_vectors.append(pos_feats)
                # 为每个正样本添加相同的蛋白质上下文
                protein_contexts.extend([protein_ctx] * len(pos_feats))

        if not positive_vectors:
            print("No positive samples available for training - skipping diffusion model training")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0).to(self.device)
        protein_contexts = torch.stack(protein_contexts).to(self.device)

        data_size = full_pos_data.size(0)
        print(f"Total positive samples: {data_size}, starting diffusion model training")

        # 特征归一化
        self.mean = full_pos_data.mean(dim=0)
        self.std = full_pos_data.std(dim=0) + 1e-8
        full_pos_data = (full_pos_data - self.mean) / self.std

        # 训练循环
        for epoch in range(epochs):
            perm = torch.randperm(data_size, device=self.device)
            losses = []
            for i in range(0, data_size, batch_size):
                idx = perm[i:i + batch_size]
                batch = full_pos_data[idx]
                ctx_batch = protein_contexts[idx]

                t = self.diffusion.sample_timesteps(batch.size(0))
                eps = torch.randn_like(batch)

                # 计算alpha_bar
                sqrt_ab = torch.sqrt(self.diffusion.alpha_bars[t])[:, None]
                sqrt_1m_ab = torch.sqrt(1 - self.diffusion.alpha_bars[t])[:, None]

                # 添加噪声
                x_t = sqrt_ab * batch + sqrt_1m_ab * eps

                # 预测噪声
                eps_pred = self.predictor(x_t, t, ctx_batch)

                # 计算损失
                loss = F.mse_loss(eps_pred, eps)

                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

                self.optimizer.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            torch.cuda.empty_cache()

    def generate_positive_sample(self, protein_context, num_samples=100):
        """生成正样本"""
        if not self.has_positive_samples:
            print("No positive samples for training - generating random samples")
            return torch.randn(num_samples, self.input_dim).cpu().detach().numpy()

        with torch.no_grad():
            x_t = torch.randn(num_samples, self.input_dim).to(self.device)

            for t in reversed(range(self.T)):
                t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                beta = self.diffusion.betas[t]
                alpha = 1 - beta
                ab = self.diffusion.alpha_bars[t]

                # 预测噪声
                eps_pred = self.predictor(x_t, t_tensor, protein_context.repeat(num_samples, 1))

                # 更新x_t
                coeff1 = 1 / torch.sqrt(alpha)
                coeff2 = (1 - alpha) / torch.sqrt(1 - ab)
                mean = coeff1 * (x_t - coeff2 * eps_pred)

                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(beta) * noise
                else:
                    x_t = mean

            # 反归一化
            x_t = x_t * self.std + self.mean
            return x_t.cpu().detach().numpy()