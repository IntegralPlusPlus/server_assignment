#ifndef GPU_H_
#define GPU_H_

#include "Request.h"

#include <string>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct PrefixMatch {
    PrefixMatch();
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;

class GPU{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::unordered_map<int, std::size_t> reservs;
    std::size_t reservatedBlocks;
    std::vector<std::pair<std::vector<std::string>, std::string> > prefixes;

    std::pair<std::vector<std::string>, std::string> getPrefix(std::size_t slot) const;
    MemoryBlock* getMemoryBlock(std::size_t slot) const;
};

#endif