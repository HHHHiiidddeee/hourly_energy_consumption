import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping to avoid overfitting"""
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0.0, path: str = "checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_iteration = 0
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_iteration += 1
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_iteration = self.best_iteration + self.counter + 1
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    """Training pipeline for models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if self.val_loader is not None:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                verbose=True,
                path=str(checkpoint_path)
            )
        else:
            self.early_stopping = None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        # Training metadata
        self.training_info = {
            'start_time': None,
            'end_time': None,
            'total_epochs': 0,
            'best_val_loss': float('inf'),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'early_stopping_patience': early_stopping_patience,
            'device': str(self.device)
        }

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(y_pred - y)).item()
            rmse = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()
            total_mae += mae
            total_rmse += rmse
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mae = total_mae / len(self.train_loader)
        avg_rmse = total_rmse / len(self.train_loader)
        
        return avg_loss, avg_mae, avg_rmse

    def _validate_epoch(self) -> Tuple[float, float, float]:
        """Validate for one epoch (or return NaN tuple when no val_loader)"""
        if self.val_loader is None or len(self.val_loader) == 0:
            return float("nan"), float("nan"), float("nan")
        
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(y_pred - y)).item()
                rmse = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()
                total_mae += mae
                total_rmse += rmse
        
        avg_loss = total_loss / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        avg_rmse = total_rmse / len(self.val_loader)
        
        return avg_loss, avg_mae, avg_rmse

    def train(self, num_epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model"""
        self.training_info['start_time'] = datetime.now().isoformat()

        for epoch in range(num_epochs):
            train_loss, train_mae, train_rmse = self._train_epoch()

            # Validation (returns NaN tuple if no val_loader)
            val_loss, val_mae, val_rmse = self._validate_epoch()

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_loss, self.model)

            # Update best val loss only when val_loss is numeric
            if not np.isnan(val_loss) and val_loss < self.training_info['best_val_loss']:
                self.training_info['best_val_loss'] = val_loss

            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - "
                      f"Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}")
            
            if self.early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        self.training_info['end_time'] = datetime.now().isoformat()
        self.training_info['total_epochs'] = len(self.history['train_loss'])

        # Load best model if checkpoint exists, otherwise save final model
        best_ckpt = self.checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            self.load_checkpoint(best_ckpt)
        else:
            final_path = self.checkpoint_dir / "final_model.pt"
            torch.save(self.model.state_dict(), final_path)
            print(f"No best model checkpoint found; saved final model to {final_path}")

        return self.history

    def test(self) -> Tuple[float, float, float]:
        """Test the model"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(y_pred - y)).item()
                rmse = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()
                total_mae += mae
                total_rmse += rmse
        
        avg_loss = total_loss / len(self.test_loader)
        avg_mae = total_mae / len(self.test_loader)
        avg_rmse = total_rmse / len(self.test_loader)
        
        return avg_loss, avg_mae, avg_rmse

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_pred = self.model(x)
        return y_pred.cpu().numpy()

    def save_checkpoint(self, checkpoint_path: str = None):
        """Save model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'training_info': self.training_info
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # EarlyStopping形式（state_dict のみ）と full形式（dict）の両方に対応
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            if 'training_info' in checkpoint:
                self.training_info = checkpoint['training_info']
        else:
            # state_dict のみの場合
            self.model.load_state_dict(checkpoint)
        
        print(f"Checkpoint loaded from {checkpoint_path}")

    def plot_history(self, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.history['val_mae'], label='Val MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(self.history['train_rmse'], label='Train RMSE')
        axes[1, 0].plot(self.history['val_rmse'], label='Val RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Training and Validation RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Validation Loss Zoom
        val_loss = self.history['val_loss']
        if len(val_loss) > 0:
            axes[1, 1].plot(val_loss, label='Val Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss (MSE)')
            axes[1, 1].set_title('Validation Loss (Zoomed)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"History plot saved to {save_path}")
        
        return fig, axes

    def save_config(self, config_path: str = None):
        """Save training configuration and results"""
        if config_path is None:
            config_path = self.checkpoint_dir / f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        config = {
            'model_name': self.model.__class__.__name__,
            'training_info': self.training_info,
            'final_metrics': {
                'best_val_loss': self.training_info['best_val_loss'],
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")

    def get_summary(self) -> Dict:
        """Get training summary"""
        return {
            'model_name': self.model.__class__.__name__,
            'total_epochs': self.training_info['total_epochs'],
            'start_time': self.training_info['start_time'],
            'end_time': self.training_info['end_time'],
            'best_val_loss': self.training_info['best_val_loss'],
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'device': self.training_info['device']
        }
