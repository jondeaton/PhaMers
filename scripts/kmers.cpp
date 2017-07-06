#include <stdio.h>
#include <map>
using namespace std;

// Allows for this to be called from Python with CTypes
extern "C" {
	int countKMers(char* sequence, size_t kmerLength, char* symbols, long* kmerCount);
}

// Function Declarations
void populateMap(map<char, size_t>& symbolIndexMap, char* symbols);
ssize_t calculateIndex(char* kmer, size_t kmerLength, map<char, size_t>& symbolIndexMap, size_t numSymbols, ssize_t index);
int ipow(int base, int exp);

int countKMers(char* sequence, size_t kmerLength, char* symbols, long* kmerCount) {

	size_t sequenceLength = strlen(sequence);
    size_t numSymbols = strlen(symbols);

	// Stores mapping from symbol to lexicographic index
	map<char, size_t> symbolIndexMap;
	populateMap(symbolIndexMap, symbols);
    
	// index is the lexicographic index in the kmerCount array corresponding
	// to the kmer under the sliding window. -1 indicates that there is no index
	// stored in this variable from the kmer under the previous window
	ssize_t index = -1;

	// Slide a window of size kmerLength along the sequence
	size_t maximumIndex = sequenceLength - kmerLength + 1;
	for (size_t i = 0; i < maximumIndex; i++) {
		char* kmer = sequence + i; // slide the window
		index = calculateIndex(kmer, kmerLength, symbolIndexMap, numSymbols, index);
		if (index == -1) continue; // bad character encountered
		kmerCount[index] += 1; // sote that we encountered this kmer
	}
	return 0;
}

void populateMap(map<char, size_t>& symbolIndexMap, char* symbols) {
	// lookup = {char: symbols.index(char) for char in symbols}
	// lookup.update({char.lower(): symbols.index(char) for char in symbols})
	size_t numSymbols = strlen(symbols);
    for (int i = 0; i < numSymbols; i++){
        symbolIndexMap[symbols[i]] = i;
        symbolIndexMap[tolower(symbols[i])] = i;    
    }
}

ssize_t calculateIndex(char* kmer, size_t kmerLength, map<char, size_t>& symbolIndexMap, size_t numSymbols, ssize_t index) {
	if (index == -1) {
		// Must recalculate
        // index = sum([lookup[kmer[n]] * pow(num_symbols, kmer_length - n - 1) for n in xrange(kmer_length)])
		index = 0;
        for (size_t j = 0; j < kmerLength; j++) {
			char letter = kmer[j];
			if (symbolIndexMap.find(letter) == symbolIndexMap.end()) return -1;

			size_t significance = ipow(numSymbols, kmerLength - j - 1);
			index += symbolIndexMap[kmer[j]] * significance;
		}
	} else {
		// May use previous window's index to make a quicker calculation
		// index = (index * num_symbols) % pow(num_symbols, kmer_length) + lookup[sequence[i + kmer_length - 1]]
		char letter = kmer[kmerLength - 1];
		if (symbolIndexMap.find(letter) == symbolIndexMap.end()) return -1;
		index = (index * numSymbols) % ipow(numSymbols, kmerLength) + symbolIndexMap[letter];
	}
	return index;
}

// Integer exponentiation
int ipow(int base, int exp) {
    if (base == 2) return (1 << exp);

    int result = 1;
    while (exp) {
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}
