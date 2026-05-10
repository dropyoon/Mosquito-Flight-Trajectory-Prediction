import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from concurrent.futures import ThreadPoolExecutor
from augmentation import get_rotation_matrix
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Configuration (하이퍼파라미터 및 경로 설정)
# ==========================================
class Config:
    data_dir = Path('./data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    train_labels_path = data_dir / 'train_labels.csv'
    sample_sub_path = data_dir / 'sample_submission.csv'
    
    # 모델 하이퍼파라미터
    input_size = 3    # x, y, z  (delta 모드에서도 feature dim은 동일)
    hidden_size = 64
    num_layers = 2
    output_size = 3   # 예측할 x, y, z

    # 입력 설정 (argparse로 덮어씀)
    use_delta    = False  # --input delta: 11 coords → 10 displacement vectors
    use_rotation = True   # --no-rotate: 회전 정규화 비활성화

    # 학습 설정
    batch_size = 128
    epochs = 600
    lr = 0.0001
    patience = 20
    seed = 42
    run_name = "GRU"  # wandb 실행 이름
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Dataset Definition (데이터 로더 정의)
# ==========================================
class MosquitoDataset(Dataset):
    _cache_dir = Path('./data/.cache')

    def __init__(self, file_paths, labels_df=None, is_train=True, augment_fns=None,
                 use_delta=False, use_rotation=True):
        self.is_train = is_train
        self.augment_fns = augment_fns if augment_fns is not None else []

        # ── 1. raw sequences 로드 (캐시 우선 → 병렬 I/O) ─────────────────
        self._cache_dir.mkdir(exist_ok=True)
        cache_key = f"{len(file_paths)}_{file_paths[0].stem}_{file_paths[-1].stem}"
        cache_file = self._cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            print(f"캐시 로드: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            raw = data['sequences']                    # (N, T, 3)
            self.file_ids = data['file_ids'].tolist()
        else:
            def _load(path):
                return np.loadtxt(str(path), delimiter=',', skiprows=1,
                                  usecols=(1, 2, 3), dtype=np.float32)

            n_workers = min(32, (os.cpu_count() or 1) * 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                seqs = list(tqdm(pool.map(_load, file_paths),
                                 total=len(file_paths), desc="Loading data"))

            self.file_ids = [p.stem for p in file_paths]
            raw = np.stack(seqs)                       # (N, T, 3)
            np.savez(cache_file, sequences=raw, file_ids=np.array(self.file_ids))
            print(f"캐시 저장: {cache_file}")

        # ── 2. augmentation (학습 시에만, raw에 적용) ─────────────────────
        if is_train and self.augment_fns:
            aug = []
            for seq in raw:
                for fn in self.augment_fns:
                    seq = fn(seq)
                aug.append(seq)
            raw = np.array(aug, dtype=np.float32)

        # ── 3. 회전 정규화 + 원점 이동 (벡터 연산) ───────────────────────
        N = len(raw)
        if use_rotation:
            rot_mats = np.array(
                [get_rotation_matrix(seq) for seq in raw], dtype=np.float32
            )                                          # (N, 3, 3)
            raw_rotated = np.einsum('ntj,nij->nti', raw, rot_mats)
        else:
            rot_mats    = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))  # identity
            raw_rotated = raw

        rot_last       = raw_rotated[:, -1, :]                    # (N, 3)
        sequences_norm = (raw_rotated - rot_last[:, np.newaxis, :]).astype(np.float32)

        # ── 4. delta 변환: (N, T, 3) → (N, T-1, 3) ───────────────────────
        if use_delta:
            sequences_norm = np.diff(sequences_norm, axis=1).astype(np.float32)

        self.sequences      = list(sequences_norm)
        self.last_positions = list(raw[:, -1, :].astype(np.float32))  # 원본 좌표계
        self.rot_mats       = list(rot_mats)

        # ── 5. 라벨 처리 (벡터 연산) ─────────────────────────────────────
        if is_train and labels_df is not None:
            labels_dict  = labels_df.set_index('id')[['x', 'y', 'z']].T.to_dict('list')
            target_array = np.array(
                [labels_dict[fid] for fid in self.file_ids], dtype=np.float32
            )                                          # (N, 3)
            displacement = target_array - raw[:, -1, :]              # (N, 3)
            # (target - last) @ R.T  →  einsum 'nj,nij->ni'
            self.targets = list(
                np.einsum('nj,nij->ni', displacement, rot_mats).astype(np.float32)
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if self.is_train:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return seq, target
        else:
            last_pos = torch.tensor(self.last_positions[idx], dtype=torch.float32)
            rot_mat = torch.tensor(self.rot_mats[idx], dtype=torch.float32)
            file_id = self.file_ids[idx]
            return seq, last_pos, rot_mat, file_id

# ==========================================
# 3. Model Definition (GRU 모델 정의)
# ==========================================
class MosquitoGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(MosquitoGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length=11, input_size=3)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        
        # 마지막 timestep(0ms)의 hidden state만 사용하여 예측
        out = out[:, -1, :] 
        
        # 변위(delta x, delta y, delta z) 예측
        out = self.fc(out)
        return out

# ==========================================
# 4. Training Loop (학습 루프)
# ==========================================
def train():
    set_seed(Config.seed)
    print(f"Using device: {Config.device}")

    wandb.init(
        project="DACON-2605-Mosquito-Trajectory",
        name=Config.run_name,
        config={
            "model":        "GRU",
            "input_size":   Config.input_size,
            "hidden_size":  Config.hidden_size,
            "num_layers":   Config.num_layers,
            "output_size":  Config.output_size,
            "batch_size":   Config.batch_size,
            "epochs":       Config.epochs,
            "lr":           Config.lr,
            "device":       str(Config.device),
            "use_delta":    Config.use_delta,
            "use_rotation": Config.use_rotation,
        },
    )

    # 파일 및 라벨 불러오기
    train_files = sorted(list(Config.train_dir.glob('TRAIN_*.csv')))
    train_labels = pd.read_csv(Config.train_labels_path)
    
    # 검증셋 분리 (8:2)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=Config.seed)
    
    # 학습에 적용할 augmentation 함수 목록 (원하는 함수를 추가/제거)
    augment_fns = [
        # translate_last_to_origin,
    ]

    # 데이터 로더 생성 (검증셋은 augmentation 없이)
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True,
                                    augment_fns=augment_fns,
                                    use_delta=Config.use_delta,
                                    use_rotation=Config.use_rotation)
    val_dataset   = MosquitoDataset(val_files, train_labels, is_train=True,
                                    use_delta=Config.use_delta,
                                    use_rotation=Config.use_rotation)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 모델, 손실함수, 옵티마이저 초기화
    model = MosquitoGRU(
        input_size=Config.input_size, 
        hidden_size=Config.hidden_size, 
        num_layers=Config.num_layers,
        output_size=Config.output_size
    ).to(Config.device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=Config.patience)
    
    ACC_THRESHOLD = 0.01  # 정답 인정 거리 기준 (m)
    best_val_loss = float('inf')
    best_val_dist_total = float('inf')

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        train_dist = 0.0
        train_correct = 0

        for seq, target in train_loader:
            seq, target = seq.to(Config.device), target.to(Config.device)

            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            dists = torch.norm(outputs.detach() - target, dim=1)
            train_loss += loss.item() * seq.size(0)
            train_dist += dists.sum().item()
            train_correct += (dists < ACC_THRESHOLD).sum().item()

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_dist /= n_train
        train_acc = train_correct / n_train

        # 검증
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        val_correct = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(Config.device), target.to(Config.device)
                outputs = model(seq)
                loss = criterion(outputs, target)
                dists = torch.norm(outputs - target, dim=1)
                val_loss += loss.item() * seq.size(0)
                val_dist += dists.sum().item()
                val_correct += (dists < ACC_THRESHOLD).sum().item()

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_dist /= n_val
        val_acc = val_correct / n_val

        print(f"Epoch [{epoch+1}/{Config.epochs}] "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f} | "
              f"Train Dist: {train_dist:.4f}  Val Dist: {val_dist:.4f} | "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        wandb.log({
            "epoch":         epoch + 1,
            "train/loss":    train_loss,
            "train/dist":    train_dist,
            "train/acc":     train_acc,
            "val/loss":      val_loss,
            "val/dist":      val_dist,
            "val/acc":       val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        # 성능이 개선되면 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dist_total = val_dist
            
            # 폴더 생성 및 모델 저장
            Path('model').mkdir(exist_ok=True)
            model_path = f'model/gru_{best_val_dist_total:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            # 추론을 위해 가장 최근의 베스트 모델도 저장
            torch.save(model.state_dict(), 'best_gru_model.pth')
            
            print(f"  --> Saved best model to {model_path}")
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_val_dist"] = best_val_dist_total
            wandb.summary["best_val_acc"]  = val_acc
            wandb.summary["best_epoch"]    = epoch + 1

    wandb.finish()
    return best_val_dist_total

# ==========================================
# 5. Inference / Prediction (추론 루프)
# ==========================================
def inference(best_val_dist=None):
    # 저장된 베스트 모델 불러오기
    model = MosquitoGRU(
        input_size=Config.input_size, 
        hidden_size=Config.hidden_size, 
        num_layers=Config.num_layers,
        output_size=Config.output_size
    ).to(Config.device)
    
    if best_val_dist is not None:
        model_path = f'model/gru_{best_val_dist:.4f}.pth'
    else:
        model_path = 'best_gru_model.pth'
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test 데이터셋 로드
    test_files = sorted(list(Config.test_dir.glob('TEST_*.csv')))
    test_dataset = MosquitoDataset(test_files, is_train=False,
                                   use_delta=Config.use_delta,
                                   use_rotation=Config.use_rotation)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for seq, last_pos, rot_mat, file_ids in test_loader:
            seq = seq.to(Config.device)

            # 회전 공간에서 변위 예측
            pred_rotated = model(seq).cpu()  # (B, 3)

            # 역회전: pred @ R  (학습 시 target @ R.T 로 변환했으므로 R.T의 역행렬 = R)
            pred_displacement = torch.bmm(
                pred_rotated.unsqueeze(1), rot_mat
            ).squeeze(1).numpy()  # (B, 3)

            # 최종 예측 위치 = 0ms 위치 + 역회전된 변위
            last_pos_np = last_pos.numpy()
            pred_pos = last_pos_np + pred_displacement
            
            for i in range(len(file_ids)):
                predictions.append({
                    'id': file_ids[i],
                    'x': pred_pos[i, 0],
                    'y': pred_pos[i, 1],
                    'z': pred_pos[i, 2]
                })
                
    # 제출 형식에 맞게 병합 후 저장
    pred_df = pd.DataFrame(predictions)
    sample_sub = pd.read_csv(Config.sample_sub_path)
    submission = sample_sub[['id']].merge(pred_df, on='id', how='left')
    
    Path('./result').mkdir(exist_ok=True)
    if best_val_dist is not None:
        sub_path = f'./result/gru_submission_{best_val_dist:.4f}.csv'
    else:
        sub_path = './result/gru_submission.csv'
        
    submission.to_csv(sub_path, index=False)
    print(f"Inference complete. Saved to {sub_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Mosquito Flight Trajectory GRU Training/Inference")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'],
                        help="Device: 'auto' (default), 'cpu', 'gpu'")
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'infer', 'all'],
                        help="Mode: 'train', 'infer', 'all' (default)")
    parser.add_argument('--name', type=str, default=None,
                        help="Wandb run name (default: from Config)")
    parser.add_argument('--input', type=str, default='raw', choices=['raw', 'delta'],
                        help="Input type: 'raw' (11×3 coords, default) or "
                             "'delta' (10×3 displacement vectors)")
    parser.add_argument('--no-rotate', dest='rotate', action='store_false',
                        help="Disable rotation normalization (last-step → +x axis). "
                             "Default: rotation ON")
    parser.set_defaults(rotate=True)
    args = parser.parse_args()

    # 설정 업데이트
    if args.name:
        Config.run_name = args.name
    Config.use_delta    = (args.input == 'delta')
    Config.use_rotation = args.rotate

    # 디바이스 설정
    if args.device == 'cpu':
        Config.device = torch.device('cpu')
    elif args.device == 'gpu':
        if torch.cuda.is_available():
            Config.device = torch.device('cuda')
        else:
            print("Warning: CUDA GPU not available. Falling back to CPU.")
            Config.device = torch.device('cpu')
    elif args.device == 'mps':
        if torch.backends.mps.is_available():
            Config.device = torch.device('mps')
        else:
            print("Warning: MPS not available. Falling back to CPU.")
            Config.device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            Config.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            Config.device = torch.device('mps')
        else:
            Config.device = torch.device('cpu')

    best_dist = None
    if args.mode in ['train', 'all']:
        print("--- Starting GRU Training ---")
        best_dist = train()
        
    if args.mode in ['infer', 'all']:
        print("\n--- Starting Inference ---")
        inference(best_dist)