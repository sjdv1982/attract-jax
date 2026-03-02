CXX ?= g++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -march=native -fPIC
# Always add current-directory include so that headers in include/ and
# forcefields/ can find nb_kernel.h via plain #include "nb_kernel.h".
CXXFLAGS_INTERNAL := -I.
LDFLAGS ?= -shared -fopenmp

TARGET ?= libnbkernel_fused.so
SRC := nb_kernel.cpp

HEADERS := nb_kernel.h \
    include/euler_rot.h \
    include/pose_loop.h \
    forcefields/nonbon8/lj.h \
    forcefields/nonbon8/lj_grad.h \
    forcefields/nonbon8/elec.h \
    forcefields/nonbon8/elec_grad.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_INTERNAL) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
