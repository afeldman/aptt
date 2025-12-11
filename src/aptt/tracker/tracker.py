from abc import ABC, abstractmethod
import torch
from torch import nn

class TrackerBase(ABC):
    @abstractmethod
    def update(self, box):
        pass

    @abstractmethod
    def predict(self):
        pass

class Track(TrackerBase):
    def __init__(self, track_id, initial_box, filter_type='kalman', device='cpu', feature=None):
        self.id = track_id
        self.boxes = [initial_box]
        self.age = 1
        self.time_since_update = 0
        self.active = True
        self.device = device
        self.appearance = feature

        if filter_type == 'kalman':
            self.filter = KalmanFilter(initial_box, device=device)
        elif filter_type == 'lstm':
            self.filter = LSTMTracker(device=device)
        elif filter_type == 'particle':
            self.filter = ParticleFilter(initial_box, device=device)

    def predict(self):
        return self.filter.predict()

    def update(self, box, feature=None):
        self.filter.update(box)
        self.boxes.append(box)
        self.time_since_update = 0
        self.age += 1
        if feature is not None:
            self.appearance = feature


class LSTMTracker(TrackerBase):
    def __init__(self, input_dim=4, hidden_dim=64, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, input_dim).to(self.device)
        self.history = []

    def update(self, box):
        box_tensor = torch.tensor(box, dtype=torch.float32, device=self.device)
        self.history.append(box_tensor)
        if len(self.history) > 10:
            self.history.pop(0)

    def predict(self):
        if len(self.history) < 2:
            return self.history[-1]
        input_seq = torch.stack(self.history).unsqueeze(0)  # [1, T, 4]
        output, _ = self.lstm(input_seq)
        pred = self.fc(output[:, -1, :])
        return pred.squeeze().detach()

class ParticleFilter(TrackerBase):
    def __init__(self, initial_box, num_particles=100, device='cpu'):
        self.device = torch.device(device)
        self.num_particles = num_particles
        box_tensor = torch.tensor(initial_box, dtype=torch.float32, device=self.device)
        self.particles = box_tensor.unsqueeze(0).repeat(num_particles, 1).clone()
        self.weights = torch.ones(num_particles, device=self.device) / num_particles

    def predict(self):
        noise = torch.randn_like(self.particles)  # std ggf. parametrisierbar
        self.particles += noise
        estimate = torch.sum(self.particles * self.weights[:, None], dim=0)
        return estimate

    def update(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        dists = torch.norm(self.particles[:, :2] - obs_tensor[:2], dim=1)
        self.weights = torch.exp(-0.5 * dists**2)
        self.weights /= torch.sum(self.weights) + 1e-6

class KalmanFilter(TrackerBase):
    def __init__(self, initial_box, device='cpu'):
        self.device = torch.device(device)

        x1, y1, x2, y2 = initial_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        self.state = torch.tensor([cx, cy, w, h, 0, 0, 0, 0], dtype=torch.float32, device=self.device)

        self.dt = 1.0
        self.F = torch.eye(8, device=self.device)
        for i in range(4):
            self.F[i, i+4] = self.dt

        self.H = torch.eye(4, 8, device=self.device)
        self.P = torch.eye(8, device=self.device) * 1000
        self.Q = torch.eye(8, device=self.device)
        self.R = torch.eye(4, device=self.device) * 10

        self._I = torch.eye(8, device=self.device)  # für Update

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_box()

    def update(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        z = torch.tensor([cx, cy, w, h], dtype=torch.float32, device=self.device)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)

        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (self._I - K @ self.H) @ self.P

    def _state_to_box(self):
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2])
