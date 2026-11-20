#include "GPU.h"
#include "MemoryBlock.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize):
        blocksNum(blocksNum), 
        blockSize(blockSize),
        reservatedBlocks(0)  {
    this->memory.resize(blocksNum);
}

std::size_t GPU::availableSlots() const {
    std::size_t numSlots = 0; 
    for (auto element : this->memory) {
        if (element == nullptr ||
             (*element).getRefCount() == 0) {
            numSlots++;
        } 
    }

    if (numSlots > this->reservatedBlocks) {
        return numSlots - this->reservatedBlocks;
    } else {
        return 0;
    }
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    if (memory % blockSize != 0) { // round up
        return memory / blockSize + 1;
    } else {
        return memory / blockSize;
    }
}

void GPU::reserve(std::size_t n, const Request& request) {
    int id = (&request)->getId();
    if (this->reservs.count(id)) {
        this->reservs[id] += n;
    } else {
        this->reservs[id] = n;
    }

    this->reservatedBlocks += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    int id = (&request)->getId();
    if (this->reservs[id] >= n) {
        this->reservs[id] -= n;
        this->reservatedBlocks -= n;
    } else {
        this->reservatedBlocks -= this->reservs[id];
        this->reservs[id] = 0;
    }
}