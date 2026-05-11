"""
Data augmentation functions for mosquito 3D flight trajectory.
Focuses on increasing data variety (reversing, sub-sequencing, etc.).
"""
import numpy as np


def reverse_trajectory(coords: np.ndarray) -> np.ndarray:
    """Augment by reversing the time order of the trajectory.
    Simulates a mosquito travelling the same path in the opposite direction.
    """
    return coords[::-1].copy()


def generate_subsequences(raw: np.ndarray, original_targets: np.ndarray,
                          seq_len: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """전체 시퀀스 배열에서 Forward + Reverse 서브시퀀스 증강을 수행한다.

    Args:
        raw: 원본 시퀀스 배열, shape (N, T, 3).
        original_targets: 라벨 배열, shape (N, 3).
        seq_len: GRU에 입력할 고정 시퀀스 길이 (default 11).

    Returns:
        Tuple of (augmented_raw, augmented_targets):
            - augmented_raw: shape (M, seq_len, 3)
            - augmented_targets: shape (M, 3)
    """
    aug_raw = []
    aug_targets = []

    for seq_idx, seq in enumerate(raw):
        orig_target = original_targets[seq_idx]

        # 1. Forward sub-sequences (정방향 서브시퀀스)
        for end_idx in range(1, 11):
            if end_idx == 9:
                continue  # +40ms target is unknown

            if end_idx == 10:
                target = orig_target
            else:
                target = seq[end_idx + 2]

            for start_idx in range(0, end_idx):  # len >= 2
                sub_seq = seq[start_idx:end_idx + 1]

                pad_len = seq_len - len(sub_seq)
                if pad_len > 0:
                    padded_seq = np.vstack([np.tile(sub_seq[0], (pad_len, 1)), sub_seq])
                else:
                    padded_seq = sub_seq

                aug_raw.append(padded_seq)
                aug_targets.append(target)

        # 2. Reverse sub-sequences (역방향 서브시퀀스)
        for start_idx in range(2, 11):
            target = seq[start_idx - 2]

            for end_idx in range(start_idx + 1, 11):  # len >= 2
                # 역방향으로 진행하는 시퀀스 추출
                sub_seq = seq[start_idx:end_idx + 1][::-1]

                pad_len = seq_len - len(sub_seq)
                if pad_len > 0:
                    padded_seq = np.vstack([np.tile(sub_seq[0], (pad_len, 1)), sub_seq])
                else:
                    padded_seq = sub_seq

                aug_raw.append(padded_seq)
                aug_targets.append(target)

    return np.array(aug_raw, dtype=np.float32), np.array(aug_targets, dtype=np.float32)
