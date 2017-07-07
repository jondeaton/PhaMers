#!/usr/bin/env python2.7
import ctypes
import numpy as np

sequence = "gtacactagcatgcacgtgagcggatgactagcacnnnnnagtagcacancagatcganacgtaca"
kmer_length = 4
symbols = "atgc"

kmer_module_file = "kmermodule.so"
kmer_module = ctypes.CDLL(kmer_module_file)

num_symbols = len(symbols)
kmer_count = np.zeros(pow(num_symbols, kmer_length), dtype=int)
arr = ctypes.c_void_p(kmer_count.ctypes.data)

print("Counting kmers with C++")
kmer_module.countKMers(sequence, kmer_length, symbols, arr)
print("done counting")
print kmer_count