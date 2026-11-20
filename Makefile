all: link
	bin/server

link: compile
	g++ -o bin/server bin/Main.o bin/GPU.o bin/MemoryBlock.o

compile: src/MemoryBlock.cpp src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o

clean:
	rm -rf bin/*