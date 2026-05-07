import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    batch_size = 128
    epochs = 30
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Dataset Definition (데이터 로더 정의)
# ==========================================
class MosquitoDataset(Dataset):
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx])
        if self.is_train:
            target = torch.tensor(self.targets[idx])
            return seq, target
        else:
            last_pos = torch.tensor(self.last_positions[idx])
            file_id = self.file_ids[idx]
            return seq, last_pos, file_id

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
    
    # 파일 및 라벨 불러오기
    train_files = sorted(list(Config.train_dir.glob('TRAIN_*.csv')))
    train_labels = pd.read_csv(Config.train_labels_path)
    
    # 검증셋 분리 (8:2)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    # 데이터 로더 생성
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True)
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
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        
        for seq, target in train_loader:
            seq, target = seq.to(Config.device), target.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(Config.device), target.to(Config.device)
                outputs = model(seq)
                loss = criterion(outputs, target)
                val_loss += loss.item() * seq.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{Config.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 성능이 개선되면 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gru_model.pth')
            print("  --> Saved best model")

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
        for seq, last_pos, file_ids in test_loader:
            seq = seq.to(Config.device)
            
            # 상대적인 변위 예측
            pred_displacement = model(seq).cpu().numpy()
            
            # 최종 예측 위치 = 0ms 위치 + 예측 변위
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
    parser = argparse.ArgumentParser(description="Mosquito Flight Trajectory GRU Training")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], 
                        help="Device to run on: 'auto' (default), 'cpu', or 'gpu'")
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

    print("--- Starting GRU Training ---")
    train()
    print("\n--- Starting Inference ---")
    inference()

