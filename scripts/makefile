# LALALALALA MAKEFILE
CC = gcc
CFLAGS = -shared
SOURCES = kmermodule.cpp
TARGETS = $(SOURCES:.cpp=.so)

CXX = /usr/bin/g++
LDFLAGS = -lm
CPPFLAGS = -Wall -pedantic -O3 -std=c++0x -shared

default: $(TARGETS)
	
kmermodule.so:
	$(CXX) $(CPPFLAGS) $(LDFLAGS)  -Wl,-install_name,kmermodule.so -o $@ -fPIC kmermodule.cpp 

.PHONY: kmermodule.so clean
clean:
	@rm -f $(TARGETS)