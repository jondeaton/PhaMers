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
	$(CXX) $(CPPFLAGS) $(LDFLAGS)  -Wl,-install_name,kmers.so -o $@ -fPIC $(SOURCES)

.PHONY: kmers.so clean
clean:
	@rm -f $(TARGETS)