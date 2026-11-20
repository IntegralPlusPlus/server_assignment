#include "GPU.h"
#include "MemoryBlock.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize) {
    this->blocksNum = blocksNum;
    this->blockSize = blockSize;
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

    return numSlots;
}