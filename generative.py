import torch
import numpy as np
import bz2
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

dfile = bz2.BZ2File('./xyData.bz2')
data = torch.from_numpy(np.load(dfile)).to(torch.float32)
dfile.close()
batch_size = 100


class XYDataset(Dataset):
    def __init__(self, xydata, transformation=None):
        self.xydata = xydata
        self.transformation = transformation

    def __len__(self):
        return self.xydata.shape[0]

    def __getitem__(self, idx):
        ret = self.xydata[idx, :, :, :]
        if self.transformation:
            ret = self.transformation(ret)

        return ret


trainset = XYDataset(data[:-10000, :, :, :])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)
testset = XYDataset(data[10000:, :, :, :])
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for grid data"""

    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()

        pe = torch.zeros(d_model, height, width)
        # Create 2D positional encoding
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)

        for i in range(0, d_model, 4):
            div_term = 10000 ** (i / d_model)
            pe[i, :, :] = torch.sin(y_pos / div_term)
            pe[i + 1, :, :] = torch.cos(y_pos / div_term)
            pe[i + 2, :, :] = torch.sin(x_pos / div_term)
            pe[i + 3, :, :] = torch.cos(x_pos / div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe.unsqueeze(0)


class SpatialAttention(nn.Module):
    """空间注意力机制，捕捉相邻格点间的关系"""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # 生成注意力图
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 增大潜在空间维度以提高表达能力
        self.latent_dim = 256

        # 编码器架构
        self.encoder = nn.Sequential(
            # 输入: [batch, 1, 16, 16]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
        )

        self.enc_attention1 = SpatialAttention(64)

        self.encoder_mid = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            # 现在: [batch, 128, 8, 8]
        )

        self.enc_attention2 = SpatialAttention(128)

        self.encoder_late = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            # 现在: [batch, 256, 4, 4]

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
        )

        # VAE潜在空间参数
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        # 解码器架构
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256 * 4 * 4),
            nn.LeakyReLU(0.2),
        )

        # 添加位置编码
        self.pos_encoding = PositionalEncoding2D(256, 4, 4)

        self.decoder_early = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            # 现在: [batch, 128, 8, 8]
        )

        self.dec_attention1 = SpatialAttention(128)

        self.decoder_mid = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            # 现在: [batch, 64, 16, 16]
        )

        self.dec_attention2 = SpatialAttention(64)

        self.decoder_final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            # 最终层输出角度
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # 直接输出角度，不使用激活函数
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """为了更好的收敛性初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        """将输入编码到潜在空间参数"""
        h = self.encoder(x)
        h = self.enc_attention1(h)
        h = self.encoder_mid(h)
        h = self.enc_attention2(h)
        h = self.encoder_late(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """VAE的重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """从潜在空间解码到输出角度"""
        h = self.decoder_fc(z)
        h = h.view(-1, 256, 4, 4)
        h = self.pos_encoding(h)  # 添加位置编码
        h = self.decoder_early(h)
        h = self.dec_attention1(h)
        h = self.decoder_mid(h)
        h = self.dec_attention2(h)
        return self.decoder_final(h)

    def forward(self, x):
        """训练的前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, batchSize):
        """从学习的分布中生成样本"""
        device = next(self.parameters()).device

        # 从标准正态分布采样
        z = torch.randn(batchSize, self.latent_dim).to(device)

        # 解码生成样本
        samples = self.decode(z)

        return samples


def neighbor_consistency_loss(x, lattice_size=16):
    """计算相邻格点间的一致性，鼓励低能量状态"""
    batch_size = x.shape[0]

    # 重塑为 [batch, height, width]
    x_flat = x.view(batch_size, lattice_size, lattice_size)

    # 计算与相邻格点的差异（周期性边界条件）
    diff_horizontal = x_flat - x_flat.roll(1, dims=2)
    diff_vertical = x_flat - x_flat.roll(1, dims=1)

    # 差异的均方值（倾向于相似的角度）
    consistency_loss = (diff_horizontal.pow(2) + diff_vertical.pow(2)).mean()

    return consistency_loss


def loss_function(recon_x, x, mu, logvar, energy_func=None, beta=1.0, energy_weight=0.5, neighbor_weight=0.2):
    """物理感知的损失函数，结合重建误差、KL散度、能量项和相邻一致性"""
    # 重建损失（角度数据的MSE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL散度损失，带beta权重
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 基于能量的损失
    if energy_func is not None:
        energy_val = energy_func(recon_x)
        energy_loss = energy_val.mean()
    else:
        energy_loss = torch.tensor(0.0, device=recon_x.device)

    # 相邻一致性损失
    neighbor_loss = neighbor_consistency_loss(recon_x)

    # 组合损失，带权重
    total_loss = recon_loss + beta * kl_loss + energy_weight * energy_loss + neighbor_weight * neighbor_loss

    return total_loss


def train(net, epochs=200, learning_rate=3e-4):
    """使用物理感知损失训练VAE模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # 使用Adam优化器，具有自适应学习率
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / 10)

    # 损失函数参数随时间逐渐增加
    beta_start, beta_end = 0.0, 1.0
    energy_weight_start, energy_weight_end = 0.1, 2.0
    neighbor_weight_start, neighbor_weight_end = 0.1, 0.5

    best_energy = float('inf')
    best_state = None

    for epoch in range(epochs):
        net.train()
        train_loss = 0

        # 课程学习 - 逐渐增加物理项的重要性
        progress = min(1.0, epoch / (epochs * 0.6))  # 在训练60%时达到目标值
        beta = beta_start + (beta_end - beta_start) * progress
        energy_weight = energy_weight_start + (energy_weight_end - energy_weight_start) * progress
        neighbor_weight = neighbor_weight_start + (neighbor_weight_end - neighbor_weight_start) * progress

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = net(data)

            # 计算带有物理项的损失
            loss = loss_function(
                recon_batch, data, mu, logvar,
                energy_func=energy,
                beta=beta,
                energy_weight=energy_weight,
                neighbor_weight=neighbor_weight
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪以提高稳定性
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item() / len(data):.6f}')

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch + 1} Average loss: {avg_loss:.6f}')
        print(f'       Beta: {beta:.4f}, Energy Weight: {energy_weight:.4f}, Neighbor Weight: {neighbor_weight:.4f}')

        scheduler.step()

        # 验证和样本生成
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                # 生成样本并评估能量
                samples = net.sample(1000)
                sample_energies = energy(samples)
                mean_energy = sample_energies.mean().item()
                min_energy = sample_energies.min().item()

                print(f'====> Sample Mean Energy: {mean_energy:.6f}, Best Sample Energy: {min_energy:.6f}')

                # 基于平均能量保存最佳模型
                if mean_energy < best_energy:
                    best_energy = mean_energy
                    best_state = {
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'energy': best_energy,
                    }
                    print(f'New best model with energy: {best_energy:.6f}')

    # 加载最佳模型
    if best_state is not None:
        net.load_state_dict(best_state['state_dict'])
        print(f'Loaded best model from epoch {best_state["epoch"]} with energy {best_state["energy"]:.6f}')

    return net


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         '''
#         Implement your model
#         '''
#         raise Exception("No implementation")
#
#     def sample(self, batchSize):
#         '''
#         Implement your model
#         This method is a must-have, which generate samples.
#         The return must be the generated [batchSize, 1, 16, 16] array.
#         '''
#         raise Exception("No implementation")
#         return samples
#
#     def implement_your_method_if_needed(self):
#         '''
#         Implement your model
#         '''
#         raise Exception("No implementation")


# def train(net):
#     '''
#     Train your model
#     '''
#     raise Exception("No implementation")

if __name__ == "__main__":
    net = NeuralNetwork()
    print(net)
    train(net)
    torch.save(net, 'generative.pth')
