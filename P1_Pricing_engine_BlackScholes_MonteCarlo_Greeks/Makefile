CXX := g++
# -O3            : optimisations agressives
# -march=native  : instructions spécifiques au CPU de la machine (SIMD, etc.)
# -fopenmp       : parallélisation du moteur Monte Carlo
# -Wall -Wextra  : garde-fous de compilation, pratique courante en production
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Iinclude

SRC := src/main.cpp src/BlackScholes.cpp src/MonteCarlo.cpp
OBJ := $(SRC:src/%.cpp=build/%.o)
TARGET := pricer

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -rf build $(TARGET)
