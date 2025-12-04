"""
Continual Learning Example
===========================

This example demonstrates continual learning (lifelong learning)
with the APTT framework, including:

- Learning Without Forgetting (LWF)
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks
- Task-incremental learning
- Catastrophic forgetting prevention
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from aptt.model.detection.resnet import ResNetBackbone
from aptt.heads.classification import ClassificationHead
from aptt.lightning_base.continual_learning_manager import ContinualLearningManager
from aptt.lightning_base.module import BaseLightningModule
from aptt.lightning_base.trainer import BaseTrainer
from aptt.loss.lwf import LearningWithoutForgettingLoss


class ContinualClassificationModule(BaseLightningModule):
    """Classification module with continual learning support."""
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone_depth: int = 18,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lwf_alpha: float = 1.0,
        lwf_temperature: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.backbone = ResNetBackbone(depth=backbone_depth)
        self.head = ClassificationHead(
            in_channels=512,
            num_classes=num_classes
        )
        
        # Loss functions
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.lwf_loss = LearningWithoutForgettingLoss(
            alpha=lwf_alpha,
            temperature=lwf_temperature
        )
        
        # Store old model for knowledge distillation
        self.old_model = None
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        # Cross-entropy loss on new task
        ce_loss = self.ce_loss(logits, labels)
        
        # Knowledge distillation loss to preserve old knowledge
        if self.old_model is not None:
            with torch.no_grad():
                old_logits = self.old_model(images)
            lwf_loss = self.lwf_loss(logits, old_logits)
            total_loss = ce_loss + lwf_loss
            
            self.log('train/ce_loss', ce_loss)
            self.log('train/lwf_loss', lwf_loss)
        else:
            total_loss = ce_loss
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.ce_loss(logits, labels)
        
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
            T_max=50,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }
    
    def store_old_model(self):
        """Store current model for knowledge distillation."""
        self.old_model = type(self)(
            num_classes=self.num_classes,
            backbone_depth=self.hparams.backbone_depth,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.old_model.load_state_dict(self.state_dict())
        self.old_model.eval()
        self.old_model.requires_grad_(False)
        print("   📦 Stored old model for LWF")
    
    def expand_head(self, new_classes: int):
        """Expand classification head for new classes."""
        old_classes = self.num_classes
        self.num_classes = old_classes + new_classes
        
        # Create new head
        new_head = ClassificationHead(
            in_channels=512,
            num_classes=self.num_classes
        )
        
        # Copy old weights
        with torch.no_grad():
            new_head.classifier.weight[:old_classes] = self.head.classifier.weight
            new_head.classifier.bias[:old_classes] = self.head.classifier.bias
        
        self.head = new_head
        print(f"   🔧 Expanded head: {old_classes} → {self.num_classes} classes")


def create_task_datasets(
    dataset_path: str = './data/cifar100',
    n_tasks: int = 5,
    batch_size: int = 128
):
    """Create task-incremental datasets from CIFAR-100."""
    
    # CIFAR-100 has 100 classes, split into tasks
    classes_per_task = 100 // n_tasks
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Load full dataset
    train_dataset = CIFAR100(
        root=dataset_path,
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = CIFAR100(
        root=dataset_path,
        train=False,
        download=True,
        transform=transform
    )
    
    # Split into tasks
    task_loaders = []
    
    for task_id in range(n_tasks):
        task_classes = list(range(
            task_id * classes_per_task,
            (task_id + 1) * classes_per_task
        ))
        
        # Filter train data
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if label in task_classes
        ]
        task_train = Subset(train_dataset, train_indices)
        
        # Filter val data
        val_indices = [
            i for i, (_, label) in enumerate(val_dataset)
            if label in task_classes
        ]
        task_val = Subset(val_dataset, val_indices)
        
        # Create loaders
        train_loader = DataLoader(
            task_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            task_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        task_loaders.append((train_loader, val_loader))
        
        print(f"Task {task_id + 1}: Classes {task_classes[0]}-{task_classes[-1]} "
              f"({len(train_indices)} train, {len(val_indices)} val)")
    
    return task_loaders


def main():
    print("🧠 APTT Continual Learning Example")
    print("=" * 60)
    
    # Configuration
    n_tasks = 5
    classes_per_task = 20  # CIFAR-100: 100 classes / 5 tasks
    epochs_per_task = 20
    
    print("Configuration:")
    print(f"  Tasks: {n_tasks}")
    print(f"  Classes per task: {classes_per_task}")
    print(f"  Epochs per task: {epochs_per_task}")
    print()
    
    # Prepare task datasets
    print("📦 Preparing task-incremental datasets...")
    task_loaders = create_task_datasets(n_tasks=n_tasks)
    print()
    
    # Initialize model with first task's classes
    print("🏗️  Initializing model...")
    model = ContinualClassificationModule(
        num_classes=classes_per_task,
        backbone_depth=18,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lwf_alpha=1.0,
        lwf_temperature=2.0
    )
    print()
    
    # Initialize continual learning manager
    cl_manager = ContinualLearningManager(
        model=model,
        strategy='lwf',  # Learning Without Forgetting
        importance_weight=1000.0
    )
    
    # Training loop over tasks
    task_accuracies = []
    
    for task_id, (train_loader, val_loader) in enumerate(task_loaders):
        print("=" * 60)
        print(f"📚 Task {task_id + 1}/{n_tasks}")
        print("=" * 60)
        
        # Expand model for new classes (except first task)
        if task_id > 0:
            print(f"\n🔧 Preparing for new task...")
            model.store_old_model()
            model.expand_head(classes_per_task)
            cl_manager.on_task_start(task_id)
        
        # Create trainer
        trainer = BaseTrainer(
            max_epochs=epochs_per_task,
            accelerator='auto',
            devices=1,
            precision='16-mixed',
            log_every_n_steps=50,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=[cl_manager]
        )
        
        # Train on current task
        print(f"\n🚀 Training on task {task_id + 1}...")
        trainer.fit(model, train_loader, val_loader)
        
        # Store task knowledge
        cl_manager.on_task_end(task_id)
        
        # Evaluate on all tasks seen so far
        print(f"\n📊 Evaluating on all tasks...")
        task_results = []
        
        for eval_task_id in range(task_id + 1):
            _, eval_loader = task_loaders[eval_task_id]
            results = trainer.validate(model, eval_loader, verbose=False)
            acc = results[0]['val/acc']
            task_results.append(acc)
            print(f"   Task {eval_task_id + 1}: {acc:.2%}")
        
        avg_acc = sum(task_results) / len(task_results)
        print(f"   Average: {avg_acc:.2%}")
        
        task_accuracies.append(task_results)
        print()
    
    # Final summary
    print("=" * 60)
    print("📈 CONTINUAL LEARNING SUMMARY")
    print("=" * 60)
    
    print("\nPer-task accuracy after each training phase:")
    print(f"{'Task':<8}", end='')
    for i in range(n_tasks):
        print(f"After T{i+1:<3}", end='')
    print()
    print("-" * (8 + n_tasks * 10))
    
    for task_id in range(n_tasks):
        print(f"Task {task_id + 1:<3}", end='')
        for phase_id in range(task_id, n_tasks):
            if phase_id < len(task_accuracies):
                if task_id < len(task_accuracies[phase_id]):
                    acc = task_accuracies[phase_id][task_id]
                    print(f"{acc:>9.1%}", end='')
                else:
                    print(f"{'N/A':>9}", end='')
        print()
    
    # Calculate final metrics
    final_accuracies = task_accuracies[-1]
    avg_acc = sum(final_accuracies) / len(final_accuracies)
    
    # Forgetting measure
    forgetting = []
    for task_id in range(n_tasks - 1):
        max_acc = max(task_accuracies[i][task_id] 
                     for i in range(task_id, n_tasks))
        final_acc = task_accuracies[-1][task_id]
        forgetting.append(max_acc - final_acc)
    
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0
    
    print(f"\n{'=' * 60}")
    print(f"📊 Final Average Accuracy: {avg_acc:.2%}")
    print(f"🧠 Average Forgetting: {avg_forgetting:.2%}")
    print(f"{'=' * 60}")
    
    print("\n✅ Continual learning completed!")


if __name__ == '__main__':
    main()
