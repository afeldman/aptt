"""
Image Classification Training Example
======================================

This example demonstrates how to train an image classification model
using the APTT framework with PyTorch Lightning.

Features:
- ResNet-based classifier
- ImageNet-style training
- Data augmentation
- Learning rate scheduling
- Checkpointing and logging
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from aptt.model.detection.resnet import ResNetBackbone
from aptt.heads.classification import ClassificationHead
from aptt.lightning_base.module import BaseLightningModule
from aptt.lightning_base.trainer import BaseTrainer


class ClassificationModel(torch.nn.Module):
    """Simple classification model with ResNet backbone."""
    
    def __init__(self, num_classes: int = 10, backbone_depth: int = 18):
        super().__init__()
        self.backbone = ResNetBackbone(depth=backbone_depth)
        # ResNet18 outputs 512 channels
        self.head = ClassificationHead(
            in_channels=512,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Use the last feature map
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)


class ClassificationModule(BaseLightningModule):
    """Lightning module for classification training."""
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone_depth: int = 18,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ClassificationModel(
            num_classes=num_classes,
            backbone_depth=backbone_depth
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


def prepare_data(data_dir: str = './data', batch_size: int = 128):
    """Prepare CIFAR-10 dataset with augmentation."""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training loop."""
    
    print("🎯 APTT Classification Training Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_classes': 10,
        'backbone_depth': 18,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'max_epochs': 100,
        'data_dir': './data/cifar10'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Prepare data
    print("📦 Preparing CIFAR-10 dataset...")
    train_loader, val_loader = prepare_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print()
    
    # Create model
    print("🏗️  Building model...")
    model = ClassificationModule(
        num_classes=config['num_classes'],
        backbone_depth=config['backbone_depth'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    print(f"   Backbone: ResNet-{config['backbone_depth']}")
    print(f"   Classes: {config['num_classes']}")
    print()
    
    # Setup trainer
    print("⚙️  Configuring trainer...")
    trainer = BaseTrainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        precision='16-mixed',  # Mixed precision for faster training
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    print("   Ready to train!")
    print()
    
    # Train
    print("🚀 Starting training...")
    print("=" * 60)
    trainer.fit(model, train_loader, val_loader)
    
    print()
    print("=" * 60)
    print("✅ Training completed!")
    print(f"   Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"   Best val/acc: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
