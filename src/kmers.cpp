#include <stdio.h>
#include <string.h>
#include <map>
using namespace std;

// Allows for this to be called from Python with CTypes
extern "C" {
	int countKMers(const char* sequence, const size_t kmerLength, const char* symbols, long* kmerCount);
}

// Function Declarations
void populateMap(map<char, size_t>& symbolIndexMap, const char* symbols);
ssize_t calculateIndex(const char* kmer, const size_t kmerLength, map<char, size_t>& symbolIndexMap, const int* significances, const size_t numSymbols, ssize_t index);
int ipow(int base, int exp);

int countKMers(const char* sequence, const size_t kmerLength, const char* symbols, long* kmerCount) {
	if (kmerLength == 0) return 0;

	size_t sequenceLength = strlen(sequence);
	if (sequenceLength < kmerLength) return 0;

    size_t numSymbols = strlen(symbols);
    if (numSymbols == 0) return 0;

	// Stores mapping from symbol to lexicographic index
	map<char, size_t> symbolIndexMap;
	populateMap(symbolIndexMap, symbols);

	// Stores the lexocographic significance of each letter in a kmer
	int* significances = new int[kmerLength + 1];
	for (size_t i = 0; i <= kmerLength; i++) significances[i] = ipow(numSymbols, i);
    
	// index is the lexicographic index in the kmerCount array corresponding
	// to the kmer under the sliding window. -1 indicates that there is no index
	// stored in this variable from the kmer under the previous window
	ssize_t index = -1;

	// Slide a window of size kmerLength along the sequence
	size_t maximumIndex = sequenceLength - kmerLength;
	for (size_t i = 0; i <= maximumIndex; i++) {
		const char* kmer = sequence + i; // slide the window
		index = calculateIndex(kmer, kmerLength, symbolIndexMap, significances, numSymbols, index);
		if (index != -1) kmerCount[index] += 1;
	}
	return 0;
}

void populateMap(map<char, size_t>& symbolIndexMap, const char* symbols) {
	// lookup = {char: symbols.index(char) for char in symbols}
	// lookup.update({char.lower(): symbols.index(char) for char in symbols})
	size_t numSymbols = strlen(symbols);
    for (size_t i = 0; i < numSymbols; i++){
        symbolIndexMap[symbols[i]] = i;
        symbolIndexMap[tolower(symbols[i])] = i;    
    }
}

ssize_t calculateIndex(const char* kmer, const size_t kmerLength, map<char, size_t>& symbolIndexMap, const int* significances, const size_t numSymbols, ssize_t index) {
	if (index == -1) {
		// Must recalculate
        // index = sum([lookup[kmer[n]] * pow(num_symbols, kmer_length - n - 1) for n in xrange(kmer_length)])
		index = 0;
        for (size_t j = 0; j < kmerLength; j++) {
			char letter = kmer[j];
			if (symbolIndexMap.find(letter) == symbolIndexMap.end()) return -1; // invalid next symbol
			index += symbolIndexMap[kmer[j]] * significances[kmerLength - j - 1];
		}
	} else {
		// May use previous window's index to make a quicker calculation
		// index = (index * num_symbols) % pow(num_symbols, kmer_length) + lookup[sequence[i + kmer_length - 1]]
		char letter = kmer[kmerLength - 1];
		if (symbolIndexMap.find(letter) == symbolIndexMap.end()) return -1;
		index = (index * numSymbols) % significances[kmerLength] + symbolIndexMap[letter];
	}
	return index;
}

// Integer exponentiation
int ipow(int base, int exp) {
	if (base == 0 || base == 1) return base;

    int result = 1;
    while (exp) {
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}
