CXX := C:/msys64/ucrt64/bin/g++.exe
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -I.

SRC := main.cpp BlackScholes.cpp AmericanPDE.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := american_pde

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) exercise_boundary.csv
