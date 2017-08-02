#include <stdio.h>
#include <map>
#include <string.h>
using namespace std;

// Allows for this to be called from Python with CTypes
extern "C" {
	int countKMers(const char* sequence, const int kmerLength, const char* symbols, long* kmerCount);
}

// Function Declarations
void populateMap(map<char, int>& symbolIndexMap, const char* symbols);
int calculateIndex(const char* kmer, const int kmerLength, map<char, int>& symbolIndexMap, const int* significances, const int numSymbols, int index);
int ipow(int base, int exp);

int countKMers(const char* sequence, const int kmerLength, const char* symbols, long* kmerCount) {
	if (kmerLength == 0) return 0;

	int sequenceLength = strlen(sequence);
	if (sequenceLength < kmerLength) return 0;

    int numSymbols = strlen(symbols);
    if (numSymbols == 0) return 0;

	// Stores mapping from symbol to lexicographic index
	map<char, int> symbolIndexMap;
	populateMap(symbolIndexMap, symbols);

	// Stores the lexocographic significance of each letter in a kmer
	int* significances = new int[kmerLength + 1];
	for (int i = 0; i <= kmerLength; i++) significances[i] = ipow(numSymbols, i);
    
	// index is the lexicographic index in the kmerCount array corresponding
	// to the kmer under the sliding window. -1 indicates that there is no index
	// stored in this variable from the kmer under the previous window
	int index = -1;

	// Slide a window of size kmerLength along the sequence
	int maximumIndex = sequenceLength - kmerLength;
	for (int i = 0; i <= maximumIndex; i++) {
		const char* kmer = sequence + i; // slide the window
		index = calculateIndex(kmer, kmerLength, symbolIndexMap, significances, numSymbols, index);
		if (index >= 0) kmerCount[index] += 1; // Valid kmer encountered
		//else i -= (index + 1); // Invalid character encountered. Advance window past it.
	}
	return 0;
}

void populateMap(map<char, int>& symbolIndexMap, const char* symbols) {
	// lookup = {char: symbols.index(char) for char in symbols}
	// lookup.update({char.lower(): symbols.index(char) for char in symbols})
	int numSymbols = strlen(symbols);
    for (int i = 0; i < numSymbols; i++){
        symbolIndexMap[symbols[i]] = i;
        symbolIndexMap[tolower(symbols[i])] = i;    
    }
}

int calculateIndex(const char* kmer, const int kmerLength, map<char, int>& symbolIndexMap, const int* significances, const int numSymbols, int index) {
	if (index < 0) {
		// Must recalculate
        // index = sum([lookup[kmer[n]] * pow(num_symbols, kmer_length - n - 1) for n in xrange(kmer_length)])
		index = 0;
		for (int j = 0; j < kmerLength; j++) {
			// char letter = kmer[j];
			if (symbolIndexMap.find(kmer[j]) == symbolIndexMap.end()) return -(j + 1); // invalid next symbol
			index += symbolIndexMap[kmer[j]] * significances[kmerLength - j - 1];
		}
	} else {
		// May use previous window's index to make a quicker calculation
		// index = (index * num_symbols) % pow(num_symbols, kmer_length) + lookup[sequence[i + kmer_length - 1]]
		// char letter = kmer[kmerLength - 1];
		if (symbolIndexMap.find(kmer[kmerLength - 1]) == symbolIndexMap.end()) return -kmerLength;
		// index = (index * numSymbols) % significances[kmerLength] + symbolIndexMap[letter];
		index = ((index % significances[kmerLength - 1]) * numSymbols) + symbolIndexMap[kmer[kmerLength - 1]];
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
