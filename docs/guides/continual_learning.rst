Continual Learning
==================

APTT unterstützt Continual Learning (lebenslanges Lernen) mit verschiedenen Strategien.

Überblick
---------

Continual Learning ermöglicht es Modellen, neue Aufgaben zu lernen, ohne das Wissen 
über frühere Aufgaben zu vergessen (Catastrophic Forgetting).

Knowledge Distillation
-----------------------

Learning without Forgetting (LwF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LwF nutzt das alte Modell als "Teacher" für neue Aufgaben:

.. code-block:: python

   from deepsuite.loss.lwf import LwF
   from deepsuite.utils.teacher import copy_teacher, freeze_teacher
   
   # Ursprüngliches Modell als Teacher kopieren
   teacher_model = copy_teacher(student_model)
   freeze_teacher(teacher_model)
   
   # LwF Loss
   lwf_loss = LwF(temperature=2.0, alpha=0.5)
   
   # Im Training Step
   def training_step(self, batch, batch_idx):
       x, y = batch
       
       # Student prediction
       student_logits = self(x)
       
       # Teacher prediction
       with torch.no_grad():
           teacher_logits = self.teacher_model(x)
       
       # Kombinierter Loss
       ce_loss = F.cross_entropy(student_logits, y)
       distill_loss = lwf_loss(student_logits, teacher_logits)
       
       total_loss = ce_loss + distill_loss
       return total_loss

Knowledge Distillation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allgemeine Distillation:

.. code-block:: python

   from deepsuite.loss.distill import Distill
   
   distill_loss = Distill(temperature=3.0, alpha=0.7)
   
   # Im BaseModule verwenden
   model = BaseModule(
       teacher_model=teacher_model,
       loss_fn=distill_loss,
       num_classes=num_classes
   )

Continual Learning Manager
---------------------------

Der ContinualLearningManager verwaltet mehrere Tasks:

.. code-block:: python

   from deepsuite.lightning_base.continual_learning_manager import ContinualLearningManager
   
   cl_manager = ContinualLearningManager(
       base_model=model,
       teacher_path="teachers"
   )
   
   # Task 1 trainieren
   cl_manager.train_task(
       task_id=1,
       datamodule=task1_datamodule,
       max_epochs=50
   )
   
   # Teacher für Task 1 speichern
   cl_manager.save_teacher(task_id=1)
   
   # Task 2 mit LwF trainieren
   cl_manager.train_task(
       task_id=2,
       datamodule=task2_datamodule,
       use_distillation=True,
       max_epochs=50
   )

Classifier Head Expansion
--------------------------

Für neue Klassen den Classifier erweitern:

.. code-block:: python

   from deepsuite.utils.head_expansion import expand_classifier
   
   # Ursprüngliches Modell hat 10 Klassen
   model = MyClassifier(num_classes=10)
   
   # Neue Klassen hinzufügen (5 zusätzliche)
   expand_classifier(model, num_old=10, num_new=5)
   
   # Modell hat jetzt 15 Klassen

Strategien
----------

Fine-Tuning mit Frozen Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Backbone einfrieren
   for param in model.backbone.parameters():
       param.requires_grad = False
   
   # Nur den Classifier trainieren
   for param in model.classifier.parameters():
       param.requires_grad = True

Progressive Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neue Spalten für neue Tasks hinzufügen:

.. code-block:: python

   class ProgressiveNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.columns = nn.ModuleList()
       
       def add_column(self, new_column):
           # Alte Spalten einfrieren
           for col in self.columns:
               for param in col.parameters():
                   param.requires_grad = False
           
           self.columns.append(new_column)

Elastic Weight Consolidation (EWC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wichtige Gewichte für alte Tasks schützen:

.. code-block:: python

   class EWC:
       def __init__(self, model, dataloader, lambda_ewc=1000):
           self.model = model
           self.lambda_ewc = lambda_ewc
           self.params = {n: p.clone().detach() 
                         for n, p in model.named_parameters()}
           self.fisher = self._compute_fisher(dataloader)
       
       def _compute_fisher(self, dataloader):
           fisher = {}
           self.model.eval()
           
           for x, y in dataloader:
               self.model.zero_grad()
               output = self.model(x)
               loss = F.cross_entropy(output, y)
               loss.backward()
               
               for n, p in self.model.named_parameters():
                   if p.grad is not None:
                       fisher[n] = p.grad.data.clone().pow(2)
           
           return fisher
       
       def penalty(self, model):
           loss = 0
           for n, p in model.named_parameters():
               loss += (self.fisher[n] * 
                       (p - self.params[n]).pow(2)).sum()
           return self.lambda_ewc * loss

Replay-basierte Methoden
~~~~~~~~~~~~~~~~~~~~~~~~

Alte Beispiele im Puffer speichern:

.. code-block:: python

   class ReplayBuffer:
       def __init__(self, capacity):
           self.capacity = capacity
           self.buffer = []
       
       def add(self, examples):
           self.buffer.extend(examples)
           if len(self.buffer) > self.capacity:
               # Älteste Beispiele entfernen
               self.buffer = self.buffer[-self.capacity:]
       
       def sample(self, n):
           import random
           return random.sample(self.buffer, min(n, len(self.buffer)))

Evaluation
----------

Metrics für Continual Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_continual_learning(model, test_loaders):
       """
       Evaluiert Modell auf allen bisherigen Tasks
       """
       accuracies = []
       
       for task_id, test_loader in enumerate(test_loaders):
           correct = 0
           total = 0
           
           model.eval()
           with torch.no_grad():
               for x, y in test_loader:
                   outputs = model(x)
                   _, predicted = outputs.max(1)
                   total += y.size(0)
                   correct += predicted.eq(y).sum().item()
           
           acc = 100. * correct / total
           accuracies.append(acc)
           print(f"Task {task_id + 1} Accuracy: {acc:.2f}%")
       
       # Average Accuracy
       avg_acc = sum(accuracies) / len(accuracies)
       print(f"Average Accuracy: {avg_acc:.2f}%")
       
       # Backward Transfer (wie sehr hilft neues Wissen alten Tasks)
       # Forward Transfer (wie gut generalisiert das Modell auf neue Tasks)
       
       return accuracies

Beispiel: Class-Incremental Learning
-------------------------------------

.. code-block:: python

   from deepsuite.lightning_base.module import BaseModule
   from deepsuite.loss.lwf import LwF
   from deepsuite.utils.teacher import copy_teacher, freeze_teacher
   from deepsuite.utils.head_expansion import expand_classifier
   
   # Phase 1: Initiales Training (Klassen 0-9)
   model = MyClassifier(num_classes=10)
   trainer = BaseTrainer(max_epochs=50)
   trainer.fit(model, datamodule=phase1_data)
   
   # Teacher speichern
   teacher_model = copy_teacher(model)
   freeze_teacher(teacher_model)
   
   # Phase 2: Neue Klassen (10-19)
   expand_classifier(model, num_old=10, num_new=10)
   
   # LwF Loss für Knowledge Distillation
   model.loss_fn = LwF(temperature=2.0, alpha=0.5)
   model.teacher_model = teacher_model
   
   # Weitertraining
   trainer = BaseTrainer(max_epochs=50)
   trainer.fit(model, datamodule=phase2_data)
   
   # Phase 3: Weitere neue Klassen (20-29)
   teacher_model = copy_teacher(model)
   freeze_teacher(teacher_model)
   
   expand_classifier(model, num_old=20, num_new=10)
   model.teacher_model = teacher_model
   
   trainer.fit(model, datamodule=phase3_data)

Best Practices
--------------

1. **Learning Rate**: Niedrigere LR für neue Tasks
2. **Regularization**: EWC oder LwF nutzen
3. **Replay Buffer**: Alte Beispiele behalten
4. **Balanced Sampling**: Gleichmäßig aus allen Tasks sampeln
5. **Evaluation**: Auf allen bisherigen Tasks testen
6. **Checkpointing**: Teachers für jeden Task speichern
7. **Feature Extraction**: Gefrorene Features als Basis nutzen
8. **Progressive Training**: Schrittweise Komplexität erhöhen
