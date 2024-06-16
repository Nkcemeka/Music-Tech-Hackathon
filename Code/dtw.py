import librosa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def compute_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def dtw_cost(mfcc1, mfcc2):
    # Compute the pairwise distance matrix
    dist_matrix = cdist(mfcc1, mfcc2, metric='euclidean')

    # Compute the cumulative cost matrix using dynamic programming
    cost_matrix = np.zeros_like(dist_matrix)
    cost_matrix[0, 0] = dist_matrix[0, 0]

    for i in range(1, dist_matrix.shape[0]):
        cost_matrix[i, 0] = dist_matrix[i, 0] + cost_matrix[i-1, 0]

    for j in range(1, dist_matrix.shape[1]):
        cost_matrix[0, j] = dist_matrix[0, j] + cost_matrix[0, j-1]

    for i in range(1, dist_matrix.shape[0]):
        for j in range(1, dist_matrix.shape[1]):
            cost_matrix[i, j] = dist_matrix[i, j] + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])

    # The cost of alignment is the bottom-right value of the cost matrix
    alignment_cost = cost_matrix[-1, -1]
    return alignment_cost

def main(file1, file2):
    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    mfcc1 = compute_mfcc(y1, sr1)
    mfcc2 = compute_mfcc(y2, sr2)

    cost = dtw_cost(mfcc1, mfcc2)
    print(f"The alignment cost between the two WAV files is: {cost}")

if __name__ == "__main__":
    # Replace with paths to your own WAV files
    file1 = '/path/to/file_1.wav'
    file2 = '/path/to/file_2.wav'
    
    main(file1, file2)

