# LALALALALA MAKEFILE
CC = gcc
CFLAGS = -shared
SOURCES = kmer-counter.cpp
TARGETS = $(SOURCES:.cpp=.so)

CXX = /usr/bin/g++
LDFLAGS = -lm
CPPFLAGS = -Wall -pedantic -O3 -std=c++0x -shared

default: $(TARGETS)
	
kmer-counter.so:
	$(CXX) $(CPPFLAGS) $(LDFLAGS)  -o $@ -fPIC $(SOURCES)

.PHONY: clean
clean:
	@rm -f $(TARGETS)
