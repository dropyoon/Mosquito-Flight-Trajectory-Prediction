import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
<<<<<<< Updated upstream
=======
import wandb
from augmentation import get_rotation_matrix
>>>>>>> Stashed changes

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
    input_size = 3    # x, y, z
    hidden_size = 64
    num_layers = 2
    output_size = 3   # 예측할 x, y, z
    
    # 학습 설정
<<<<<<< Updated upstream
    batch_size = 64
    epochs = 600
    lr = 5e-4
    patience = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
=======
    batch_size = 128
    epochs = 600
    lr = 0.0001
    device = torch.device('cpu')
>>>>>>> Stashed changes

# ==========================================
# 2. Dataset Definition (데이터 로더 정의)
# ==========================================
class MosquitoDataset(Dataset):
<<<<<<< Updated upstream
    def __init__(self, file_paths, labels_df=None, is_train=True):
        self.file_paths = file_paths
        self.is_train = is_train
        
        self.sequences = []
        self.last_positions = []
        self.file_ids = []
        
        # 파일별로 데이터를 읽어 메모리에 적재
        for path in tqdm(file_paths, desc="Loading data"):
            df = pd.read_csv(path)
            # Shape: (11, 3) -> 11 timesteps (-400ms to 0ms)
            seq = df[['x', 'y', 'z']].values.astype(np.float32)
            
            # 모델의 학습 효율을 높이기 위해 마지막 위치(0ms)를 기준으로 상대 위치(변위)로 변환
            last_pos = seq[-1].copy()
            seq_norm = seq - last_pos
            
            self.sequences.append(seq_norm)
            self.last_positions.append(last_pos)
            self.file_ids.append(path.stem)
            
        if self.is_train and labels_df is not None:
            # 라벨도 변위(Displacement)로 변환하여 Target으로 설정
            labels_dict = labels_df.set_index('id')[['x', 'y', 'z']].T.to_dict('list')
            self.targets = []
            for fid, last_pos in zip(self.file_ids, self.last_positions):
                target_pos = np.array(labels_dict[fid], dtype=np.float32)
                target_norm = target_pos - last_pos # 목표 위치까지의 변위
                self.targets.append(target_norm)
=======
    _cache_dir = Path('./data/.cache')

    def __init__(self, file_paths, labels_df=None, is_train=True, augment_fns=None):
        self.is_train = is_train
        self.augment_fns = augment_fns or []

        # ── 1. raw sequences 로드 (캐시 우선) ──────────────────────────
        self._cache_dir.mkdir(exist_ok=True)
        cache_key = f"{len(file_paths)}_{file_paths[0].stem}_{file_paths[-1].stem}"
        cache_file = self._cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            print(f"캐시 로드: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            raw = data['sequences']               # (N, T, 3)
            self.file_ids = data['file_ids'].tolist()
        else:
            # pd.read_csv 대신 np.loadtxt 사용 (파일당 ~10배 빠름)
            raw_list, file_ids = [], []
            for path in tqdm(file_paths, desc="Loading data"):
                seq = np.loadtxt(path, delimiter=',', skiprows=1,
                                 usecols=(1, 2, 3), dtype=np.float32)
                raw_list.append(seq)
                file_ids.append(path.stem)
            raw = np.stack(raw_list)              # (N, T, 3)
            np.savez(cache_file, sequences=raw, file_ids=np.array(file_ids))
            print(f"캐시 저장 완료: {cache_file}")
            self.file_ids = file_ids

        # ── 2. 정규화: 마지막 좌표를 원점으로 → 진행방향 x축 정렬 ──
        last_positions = raw[:, -1, :]                            # (N, 3)
        sequences_translated = (raw - last_positions[:, np.newaxis, :]).astype(np.float32)

        # 샘플별 회전행렬 계산 후 시퀀스에 적용
        # sequences_translated[n] @ R[n].T  →  einsum 'ntj,nij->nti'
        rot_mats = np.array(
            [get_rotation_matrix(seq) for seq in sequences_translated], dtype=np.float32
        )                                                         # (N, 3, 3)
        sequences_norm = np.einsum('ntj,nij->nti', sequences_translated, rot_mats).astype(np.float32)

        self.sequences = list(sequences_norm)
        self.last_positions = list(last_positions.astype(np.float32))
        self.rot_mats = list(rot_mats)                            # 추론 시 역회전에 사용

        # ── 3. 라벨 처리 ────────────────────────────────────────────────
        if is_train and labels_df is not None:
            labels_dict = labels_df.set_index('id')[['x', 'y', 'z']].to_dict('index')
            target_array = np.array(
                [[labels_dict[fid]['x'], labels_dict[fid]['y'], labels_dict[fid]['z']]
                 for fid in self.file_ids], dtype=np.float32
            )
            target_displaced = (target_array - last_positions).astype(np.float32)  # (N, 3)
            # 시퀀스와 동일한 R로 타겟도 회전해야 좌표계가 일치함
            # target_displaced[n] @ R[n].T  →  einsum 'nj,nij->ni'
            target_rotated = np.einsum('nj,nij->ni', target_displaced, rot_mats).astype(np.float32)
            self.targets = list(target_rotated)
>>>>>>> Stashed changes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx])
        if self.is_train:
            target = torch.tensor(self.targets[idx])
            return seq, target
        else:
            last_pos = torch.tensor(self.last_positions[idx])
            rot_mat = torch.tensor(self.rot_mats[idx])  # (3, 3) — 역회전용
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
    print(f"Using device: {Config.device}")
<<<<<<< Updated upstream
    
    # wandb 초기화
    wandb.init(
        project="DACON-2605-Mosquito",
        name="GRU",
        config={
            "input_size": Config.input_size,
            "hidden_size": Config.hidden_size,
            "num_layers": Config.num_layers,
            "batch_size": Config.batch_size,
            "epochs": Config.epochs,
            "lr": Config.lr,
            "patience": Config.patience
        }
    )
    
=======

    wandb.init(
        project="mosquito-trajectory",
        config={
            "model":       "GRU",
            "input_size":  Config.input_size,
            "hidden_size": Config.hidden_size,
            "num_layers":  Config.num_layers,
            "output_size": Config.output_size,
            "batch_size":  Config.batch_size,
            "epochs":      Config.epochs,
            "lr":          Config.lr,
            "device":      str(Config.device),
        },
    )

>>>>>>> Stashed changes
    # 파일 및 라벨 불러오기
    train_files = sorted(list(Config.train_dir.glob('TRAIN_*.csv')))
    train_labels = pd.read_csv(Config.train_labels_path)
    
    # 검증셋 분리 (8:2)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
<<<<<<< Updated upstream
    # 데이터 로더 생성
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True)
=======
    # 학습에 적용할 augmentation 함수 목록 (원하는 함수를 추가/제거)
    augment_fns = [
        # translate_last_to_origin,
    ]

    # 데이터 로더 생성 (검증셋은 augmentation 없이)
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True, augment_fns=augment_fns)
>>>>>>> Stashed changes
    val_dataset = MosquitoDataset(val_files, train_labels, is_train=True)
    
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

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
<<<<<<< Updated upstream
        train_correct = 0
        
=======
        train_dist = 0.0
        train_correct = 0

>>>>>>> Stashed changes
        for seq, target in train_loader:
            seq, target = seq.to(Config.device), target.to(Config.device)

            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            dists = torch.norm(outputs.detach() - target, dim=1)
            train_loss += loss.item() * seq.size(0)
<<<<<<< Updated upstream
            
            # 예측값과 실제값의 거리 계산 (0.01m 이하인 경우 정답)
            distances = torch.linalg.norm(outputs - target, dim=1)
            train_correct += (distances <= 0.01).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # 검증
        model.eval()
        val_loss = 0.0
=======
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
>>>>>>> Stashed changes
        val_correct = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(Config.device), target.to(Config.device)
                outputs = model(seq)
                loss = criterion(outputs, target)
                dists = torch.norm(outputs - target, dim=1)
                val_loss += loss.item() * seq.size(0)
<<<<<<< Updated upstream
                
                # 예측값과 실제값의 거리 계산 (0.01m 이하인 경우 정답)
                distances = torch.linalg.norm(outputs - target, dim=1)
                val_correct += (distances <= 0.01).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # 현재 학습률 가져오기
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{Config.epochs}], LR: {current_lr:.6f}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        # wandb 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })
        
        # 스케줄러 업데이트 (train_loss 기준)
        scheduler.step(train_loss)
        
=======
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
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/dist": train_dist,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/dist":   val_dist,
            "val/acc":    val_acc,
        })

>>>>>>> Stashed changes
        # 성능이 개선되면 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gru_model.pth')
            print("  --> Saved best model")
<<<<<<< Updated upstream
            
=======
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_val_dist"] = val_dist
            wandb.summary["best_val_acc"]  = val_acc

>>>>>>> Stashed changes
    wandb.finish()

# ==========================================
# 5. Inference / Prediction (추론 루프)
# ==========================================
def inference():
    # 저장된 베스트 모델 불러오기
    model = MosquitoGRU(
        input_size=Config.input_size, 
        hidden_size=Config.hidden_size, 
        num_layers=Config.num_layers,
        output_size=Config.output_size
    ).to(Config.device)
    
    model.load_state_dict(torch.load('best_gru_model.pth'))
    model.eval()
    
    # Test 데이터셋 로드
    test_files = sorted(list(Config.test_dir.glob('TEST_*.csv')))
    test_dataset = MosquitoDataset(test_files, is_train=False)
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
    submission.to_csv('./result/gru_submission.csv', index=False)
    print("Inference complete. Saved to ./result/gru_submission.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Mosquito Flight Trajectory GRU Training/Inference")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], 
                        help="Device to run on: 'auto' (default), 'cpu', or 'gpu'")
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'infer', 'all'],
                        help="Mode to run: 'train', 'infer', or 'all' (default)")
    args = parser.parse_args()

    # 디바이스 설정
    if args.device == 'cpu':
        Config.device = torch.device('cpu')
    elif args.device == 'gpu':
        if torch.cuda.is_available():
            Config.device = torch.device('cuda')
        else:
            print("Warning: GPU requested but not available. Falling back to CPU.")
            Config.device = torch.device('cpu')
    else:
        Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode in ['train', 'all']:
        print("--- Starting GRU Training ---")
        train()
        
    if args.mode in ['infer', 'all']:
        print("\n--- Starting Inference ---")
        inference()

