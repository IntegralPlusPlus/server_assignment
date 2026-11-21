all: link

run: link
	./bin/server config.json
link: compile
	g++ -o bin/server bin/Main.o bin/GPU.o bin/MemoryBlock.o bin/Request.o bin/Model.o

compile: src/MemoryBlock.cpp src/Main.cpp src/GPU.cpp src/Model.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Request.cpp -o bin/Request.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o

clean:
	rm -rf bin/*