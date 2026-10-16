import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import Detector


class _AE(nn.Module):
    def __init__(self, bottleneck: int = 4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, bottleneck), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderDetector(Detector):
    name = "autoenc"

    def __init__(self, threshold: float = 2.0, bottleneck: int = 4,
                 epochs: int = 300, batch_size: int = 64,
                 lr: float = 1e-3, seed: int = 42, patience: int = 20):
        torch.manual_seed(seed)
        self.threshold = threshold
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.model = _AE(bottleneck)
        self.mean_ = None
        self.std_ = None
        self._err_mean = None
        self._err_std = None
        self.history = []   # loss per epoch

    def _errors(self, x_scaled: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(x_scaled, dtype=torch.float32).view(-1, 1)
            recon = self.model(t).view(-1)
            err = (recon - t.view(-1)).abs().numpy()
        return err

    def fit(self, values):
        x = np.asarray(values, dtype=float)
        self.mean_ = float(x.mean())
        self.std_  = float(x.std()) + 1e-9
        xn = (x - self.mean_) / self.std_

        dataset = TensorDataset(torch.tensor(xn, dtype=torch.float32).view(-1, 1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            self.history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        errs = self._errors(xn)
        self._err_mean = float(errs.mean())
        self._err_std  = float(errs.std()) + 1e-9
        return self

    def score(self, x: float) -> float:
        xn = (x - self.mean_) / self.std_
        err = float(self._errors(np.array([xn]))[0])
        return (err - self._err_mean) / self._err_std