#include <iostream>
#include "Server.h"
#include "MemoryBlock.h"

using namespace std;

int main(int argc, char** argv){
    /*if(argc < 2){
        cerr << "usage: " << argv[0] << " <config.json>" << endl;
        return 1;
    }
    Server s(argv[1]);
    s.serve();*/

    KVBlock b(5);
    b.writeToBlock(1, 5);
    
    KVBlock a(b);
    
    b.writeToBlock(1, 7);

    cout << a.readFromBlock(1) << endl;

    return 0;
}
