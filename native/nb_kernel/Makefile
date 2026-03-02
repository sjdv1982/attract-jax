CXX ?= g++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -march=native -fPIC
LDFLAGS ?= -shared -fopenmp

TARGET ?= libnbkernel_fused.so
SRC := nb_kernel.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) nb_kernel.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
