# LALALALALA MAKEFILE
CC = gcc
CFLAGS = -shared
SOURCES = kmers.cpp
TARGETS = $(SOURCES:.cpp=.so)

CXX = /usr/bin/g++
LDFLAGS = -lm
CPPFLAGS = -Wall -pedantic -O3 -std=c++0x -shared

default: $(TARGETS)
	
kmers.so:
	$(CXX) $(CPPFLAGS) $(LDFLAGS)  -o $@ -fPIC $(SOURCES)

.PHONY: clean
clean:
	@rm -f $(TARGETS)