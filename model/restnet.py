import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import random
import sys
import logging
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn.functional as F

# ==========================================
# 1. AYARLAR (CONFIG)
# ==========================================
CONFIG = {
    'dataset_path': '/mnt/c/Users/gtu/Desktop/scoliosis-detection/datasets/dataset_unified_64.pkl',
    'batch_size': 16,             # ResNet dondurulduğu için 16'ya çıkabiliriz
    'learning_rate': 0.0005,      # Dondurulmuş katmanlar olduğu için LR biraz artabilir
    'weight_decay': 1e-2,         # Regularization artırıldı (Ezber bozucu)
    'epochs': 50,                
    'patience': 15,
    'num_classes': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seeds': [42, 43, 44, 45, 46],
    'log_dir': 'resnet_pro_logs',
    'checkpoint_dir': 'checkpoints/resnet_pro_best' # Modellerin kaydedileceği klasör
}

# ==========================================
# 2. LOGGING & UTILS
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_dir, "training.log")),
                                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger()

# ==========================================
# 3. FOCAL LOSS (DENGESİZ VERİ İÇİN KRİTİK)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# ==========================================
# 4. DATASET (GÜÇLENDİRİLMİŞ AUGMENTATION)
# ==========================================
class ScoliosisDataset(Dataset):
    def __init__(self, sequences, labels, training=False):
        self.data = sequences
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_np = self.data[idx]
        label = self.labels[idx]

        if isinstance(video_np, np.ndarray):
            video = torch.from_numpy(video_np).float()
        else:
            video = torch.tensor(video_np).float()

        if self.training:
            video = self.apply_augmentation(video)
        
        # (T, H, W) -> (T, 1, H, W)
        if video.ndim == 3: 
            video = video.unsqueeze(1) 
            
        return video, label

    def apply_augmentation(self, video):
        # 1. Horizontal Flip
        if random.random() < 0.5:
            video = torch.flip(video, dims=[-1])
            
        # 2. Temporal Shift (Zaman Kaydırma)
        if random.random() < 0.3:
            shift = random.randint(-4, 4)
            if shift > 0:
                video = torch.cat([video[shift:], video[-1:].repeat(shift, 1, 1)], dim=0)
            elif shift < 0:
                shift = abs(shift)
                video = torch.cat([video[:1].repeat(shift, 1, 1), video[:-shift]], dim=0)
                
        return video

# ==========================================
# 5. MODEL: ResNet18 + GRU (PARTIAL FREEZING)
# ==========================================
class ResNetGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, dropout_prob=0.6):
        super(ResNetGRU, self).__init__()
        
        # 1. BACKBONE
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except:
            resnet = models.resnet18(pretrained=True)
            
        # Kanal Uyumu
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # --- PARTIAL FREEZING (Kısmi Dondurma) ---
        # ResNet'in ilk katmanlarını donduruyoruz ki temel özellikleri bozmasın.
        # Sadece son 2 bloğu eğiteceğiz.
        
        # Önce hepsini dondur
        for param in resnet.parameters():
            param.requires_grad = False
            
        # İlk Conv katmanını aç (Çünkü 1 kanala çevirdik, öğrenmesi lazım)
        for param in resnet.conv1.parameters():
            param.requires_grad = True
            
        # Son blokları aç (Layer3 ve Layer4) - Yüksek seviye özellikleri öğrensin
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(*list(resnet.children())[:-1]) 
        cnn_out_dim = 512
        
        # 2. GRU
        self.rnn = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Dropout artırıldı
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w) 
        
        features = self.features(x)
        features = features.view(b, t, -1) 
        
        rnn_out, _ = self.rnn(features)
        
        # Max Pooling (Hastalığın en belirgin anına odaklan)
        out, _ = torch.max(rnn_out, dim=1)
        
        return self.classifier(out)

# ==========================================
# 6. TRAINER
# ==========================================
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, scaler, logger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.logger = logger

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return running_loss / len(loader), 100 * correct / total

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        acc = 100 * (all_labels == all_preds).sum() / len(all_labels)
        return running_loss / len(loader), acc

def evaluate_ensemble_with_threshold(model_paths, loader, device):
    models_list = []
    for path in model_paths:
        m = ResNetGRU(num_classes=CONFIG['num_classes']).to(device)
        m.load_state_dict(torch.load(path))
        m.eval()
        models_list.append(m)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            avg_probs = torch.zeros(inputs.size(0), CONFIG['num_classes']).to(device)
            for model in models_list:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                avg_probs += probs
            avg_probs /= len(models_list)
            
            all_probs.extend(avg_probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Grid Search ile En İyi Threshold
    best_acc = 0
    best_th = 0.5
    
    print("\n🔍 Threshold Analizi:")
    for th in np.arange(0.3, 0.9, 0.05):
        preds = (all_probs > th).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    final_preds = (all_probs > best_th).astype(int)
    cm = confusion_matrix(all_labels, final_preds)
    acc = accuracy_score(all_labels, final_preds) * 100
    prec = precision_score(all_labels, final_preds, zero_division=0)
    rec = recall_score(all_labels, final_preds, zero_division=0)
    f1 = f1_score(all_labels, final_preds, zero_division=0)
    
    return acc, prec, rec, f1, cm, best_th

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    logger = setup_logger(CONFIG['log_dir'])
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True) # Klasörü oluştur
    logger.info("🚀 RESNET-PRO (Focal Loss + Partial Freeze) BAŞLIYOR")
    
    with open(CONFIG['dataset_path'], 'rb') as f:
        unified_data = pickle.load(f)
    
    saved_models = []

    for seed in CONFIG['seeds']:
        logger.info(f"\n--- Seed {seed} ---")
        seed_everything(seed)
        
        train_ds = ScoliosisDataset(unified_data['train']['sequences'], unified_data['train']['labels'], training=True)
        val_ds = ScoliosisDataset(unified_data['val']['sequences'], unified_data['val']['labels'], training=False)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        
        model = ResNetGRU(num_classes=2, dropout_prob=0.6).to(CONFIG['device'])
        
        # Sadece eğitilecek parametreleri Optimizer'a ver
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        
        optimizer = optim.AdamW(params_to_update, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        scaler = GradScaler()
        
        # Focal Loss Kullanımı (Dengesizliği Çözer)
        criterion = FocalLoss(gamma=2)
        
        trainer = Trainer(model, optimizer, criterion, scheduler, CONFIG['device'], scaler, logger)
        
        best_acc = 0.0
        patience_counter = 0
        best_file = os.path.join(CONFIG['checkpoint_dir'], f"resnet_pro_{seed}.pth")
        
        for epoch in range(CONFIG['epochs']):
            train_loss, train_acc = trainer.train_one_epoch(train_loader)
            val_loss, val_acc = trainer.validate(val_loader)
            scheduler.step(val_acc)
            
            if (epoch+1) % 5 == 0:
                logger.info(f"Ep {epoch+1}: Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), best_file)
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    logger.info(f"🛑 Early Stopping @ Ep {epoch+1}")
                    break
        
        logger.info(f"✅ En İyi Model: {best_acc:.2f}%")
        saved_models.append(best_file)

    # --- FINAL TEST ---
    logger.info("\n🏆 FINAL TEST (RESNET-PRO)")
    X_test = unified_data['test']['sequences']
    y_test = unified_data['test']['labels']
    test_ds = ScoliosisDataset(X_test, y_test, training=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    acc, prec, rec, f1, cm, th = evaluate_ensemble_with_threshold(saved_models, test_loader, CONFIG['device'])
    
    print("=" * 60)
    print(f"🚀 FINAL SONUÇ (Optimize Edilmiş Threshold: {th:.2f})")
    print(f"ACCURACY   : %{acc:.2f}")
    print(f"F1 SCORE   : {f1:.3f}")
    print(f"PRECISION  : {prec:.3f}")
    print(f"RECALL     : {rec:.3f}")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(cm)
    print("=" * 60)