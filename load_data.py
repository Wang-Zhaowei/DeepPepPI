import numpy as np
import re
import math


def padding(fea_mat, max_len):
    seq_len = fea_mat.shape[0]
    pad_len = max_len - seq_len
    pad_mat = np.zeros((pad_len, fea_mat.shape[1]))
    padded_mat = np.vstack((fea_mat, pad_mat))
    return padded_mat


def extract_pept_feat(file_path):
    emb_dict = np.load(file_path, allow_pickle=True).item()
    return emb_dict


def get_korder_base(chars, k):
    korder_chars = []
    chars_len = len(chars)
    for i in range(0, chars_len**k):
        n = i
        bases = ''
        for j in range(0, k):
            base = chars[n%chars_len]
            n=n//chars_len
            bases += base
        korder_chars.append(bases[::-1])
    return korder_chars


def build_prot_mat(seq, str, k):
    seq_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    str_chars = ['C', 'H', 'E']
    seq = re.sub(r"[UZOB]", "X", seq)
    seq_korder = get_korder_base(seq_chars, k)
    str_korder = get_korder_base(str_chars, k)
    row = len(seq_chars)**k
    col = len(str_chars)**k
    prot_fea_mat = np.zeros(shape=(row, col))
    
    for i in range(0, len(seq)-k+1):
        s, e = i, i+k
        seq_base = seq[s:e]
        str_base = str[s:e]
        if 'X' in seq_base:
            prot_fea_mat[:][str_korder.index(str_base)] += 1/len(seq_korder)
        else:
            row_index = seq_korder.index(seq_base)
            col_index = str_korder.index(str_base)
            prot_fea_mat[row_index][col_index] += 1
    return prot_fea_mat


def load_sequences(file_path):
    seq_dict = {}
    with open(file_path, 'r') as rf:
        seq = ''
        for line in rf:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
            else:
                seq = line.upper()
                seq_dict[name] = seq
    return seq_dict


def extract_prot_feat(seq_file, str_file, order):
    seq_dict = load_sequences(seq_file)
    str_dict = load_sequences(str_file)
    mat_dict = {}
    for prot in seq_dict.keys():
        seq = seq_dict[prot]
        str = str_dict[prot]
        mat_dict[prot] = build_prot_mat(seq, str, order)
    return mat_dict


def load_pairs(file_path):
    pairs = []
    with open(file_path, 'r') as rf:
        for pair in rf:
            pept, prot = pair.split()
            pairs.append([pept.strip(), prot.strip()])
    return pairs


def load_dataset(pept_emb_dict, prot_mat_dict, posi_pairs, nega_pairs):
	pept_emb_list = []
	max_len = 100
	prot_mat_list = []

	label = []
	for pair in posi_pairs:
		pept, prot = pair[0], pair[1]
		pept_emb = padding(pept_emb_dict[pept], max_len)
		pept_emb_list.append(pept_emb)
		prot_mat_list.append(prot_mat_dict[prot].T)
		label.append([0, 1])
	for pair in nega_pairs:
		pept, prot = pair[0], pair[1]
		pept_emb = padding(pept_emb_dict[pept], max_len)
		pept_emb_list.append(pept_emb)
		prot_mat_list.append(prot_mat_dict[prot].T)
		label.append([1, 0])
	return np.array(pept_emb_list), np.array(prot_mat_list), np.array(label)