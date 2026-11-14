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

    KVBlock mb(5);
    
    cout << mb.readFromBlock(1) << endl;

    return 0;
}
