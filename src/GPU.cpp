#include "GPU.h"
#include "MemoryBlock.h"

PrefixMatch::PrefixMatch():
    matchedTokens(0) {
}

GPU::GPU(std::size_t blocksNum, std::size_t blockSize):
        blocksNum(blocksNum), 
        blockSize(blockSize),
        reservatedBlocks(0),
        cache_hits(0),
        evictions(0) {
    this->memory.resize(blocksNum);
    this->prefixes.resize(blocksNum);
}

GPU::GPU(const GPU& other):
        blocksNum(other.blocksNum),
        blockSize(other.blockSize),
        reservs(other.reservs),
        reservatedBlocks(other.reservatedBlocks),
        cache_hits(other.cache_hits),
        evictions(other.evictions) {
    this->memory.resize(this->blocksNum);
    this->prefixes.resize(this->blocksNum);

    for (std::size_t slot = 0; slot < this->blocksNum; ++slot) {
        if (other.memory[slot] != nullptr) {
            this->memory[slot] = other.memory[slot]->clone();
            this->memory[slot]->setRefCount(other.memory[slot]->getRefCount());
        } else {
            this->memory[slot] = nullptr;
        }

        this->prefixes[slot] = other.prefixes[slot];
    }
}

GPU& GPU::operator=(const GPU& other) {
    if (this == &other) {
        return *this;
    }

    this->blocksNum = other.blocksNum;
    this->blockSize = other.blockSize;
    this->reservs = other.reservs;
    this->reservatedBlocks = other.reservatedBlocks;
    this->cache_hits = other.cache_hits;
    this->evictions = other.evictions;

    for (auto* block : this->memory) {
        delete block;
    }
    
    this->memory.clear();

    this->memory.resize(this->blocksNum);
    this->prefixes.resize(this->blocksNum);

    for (std::size_t slot = 0; slot < this->blocksNum; ++slot) {
        if (other.memory[slot] != nullptr) {
            this->memory[slot] = other.memory[slot]->clone();
            this->memory[slot]->setRefCount(other.memory[slot]->getRefCount());
        } else {
            this->memory[slot] = nullptr;
        }

        this->prefixes[slot] = other.prefixes[slot];
    }

    return *this;
}

GPU::GPU(GPU&& other):
        blocksNum(other.blocksNum),
        blockSize(other.blockSize),
        reservs(other.reservs),
        reservatedBlocks(other.reservatedBlocks),
        cache_hits(other.cache_hits),
        evictions(other.evictions) {
    this->memory.resize(this->blocksNum);
    this->prefixes.resize(this->blocksNum);

    for (std::size_t slot = 0; slot < this->blocksNum; ++slot) {
        this->memory[slot] = other.memory[slot];
        this->prefixes[slot] = other.prefixes[slot];
        
        other.memory[slot] = nullptr;
        other.prefixes[slot] = std::pair<std::vector<std::string>, std::string>();
    }

    other.memory.clear();
    other.prefixes.clear();
    other.blocksNum = 0;
    other.reservs = std::unordered_map<int, std::size_t>();
    other.reservatedBlocks = 0;
}

GPU& GPU::operator=(GPU&& other) {
    if (this == &other) {
        return *this;
    }

    this->blocksNum = other.blocksNum;
    this->blockSize = other.blockSize;
    this->reservs = other.reservs;
    this->reservatedBlocks = other.reservatedBlocks;
    this->cache_hits = other.cache_hits;
    this->evictions = other.evictions;

    for (auto* block : this->memory) {
        delete block;
    }

    this->memory.resize(this->blocksNum);
    this->prefixes.resize(this->blocksNum);

    for (std::size_t slot = 0; slot < this->blocksNum; ++slot) {
        this->memory[slot] = other.memory[slot];
        this->prefixes[slot] = other.prefixes[slot];
        
        other.memory[slot] = nullptr;
        other.prefixes[slot] = std::pair<std::vector<std::string>, std::string>();
    }

    other.memory.clear();
    other.prefixes.clear();
    other.blocksNum = 0;
    other.reservs = std::unordered_map<int, std::size_t>();
    other.reservatedBlocks = 0;

    return *this;
}

GPU::~GPU() {
    for (auto memoryBlock : this->memory) {
        delete memoryBlock;
    }
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
    if (this->reservs[id] > n) {
        this->reservs[id] -= n;
        this->reservatedBlocks -= n;
    } else {
        this->reservatedBlocks -= this->reservs[id];
        this->reservs[id] = 0;
    }
}

std::size_t GPU::allocateBlock(const Request& request) {
    std::size_t i = 0, slot = this->blocksNum, lowestIdle = this->blocksNum;
    int id = request.getId();
    
    for (auto element : this->memory) {
        if (element == nullptr) {
            slot = i;
            break;
        } else if ((*element).getRefCount() == 0 && lowestIdle == this->blocksNum) {
            lowestIdle = i;
        }

        i++;
    }

    if (slot == this->blocksNum) {
        slot = lowestIdle;
        delete this->memory[slot];
        this->prefixes[slot] = {};
        this->evictions++;
    }

    request.allocateOneBlock(*this, slot);
    this->reservatedBlocks--;
    this->reservs[id]--;

    return slot;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    this->memory[slot] = new KVBlock(this->blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    this->memory[slot] = new EmbeddingWorkspace(this->blockSize);
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    for (auto prefix : prefixes) {
        if (prefix.first == prefixTokens && prefix.second == modelId) {
            return;
        }
    }

    prefixes[block] = std::make_pair(prefixTokens, modelId);
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const {
    PrefixMatch prefix;
    std::size_t tokens = this->blockSize / embeddingSize;
    std::vector<std::size_t> slotsPerTokens(promptTokens.size() / tokens + 1);

    for (std::size_t block = 0; block < this->blocksNum; ++block) {
        if (this->prefixes[block].second == modelId 
            && this->prefixes[block].first.size() != 0
            && this->prefixes[block].first.size() <= promptTokens.size()) {
            
            std::size_t degree = 0;
            while ((degree + 1) * tokens <= this->prefixes[block].first.size()) {
                bool updateDegree = true;
                for (std::size_t i = 0; i < tokens; ++i) {
                    if (this->prefixes[block].first[degree * tokens + i] != promptTokens[degree * tokens + i]) {
                        updateDegree = false;
                        break;
                    }
                }
                
                if (!updateDegree) break;

                degree++;
            }

            if (degree * tokens == this->prefixes[block].first.size()) {
                slotsPerTokens[degree] = block + 1;
            }
        }
    }

    for (std::size_t i = 1; i < slotsPerTokens.size(); ++i) {
        if (slotsPerTokens[i] == 0) {
            break;
        } else {
            prefix.matchedTokens = i * tokens;
            prefix.blockIds.push_back(slotsPerTokens[i] - 1);
        }
    }

    return prefix;
}

void GPU::acquireCacheHits(const PrefixMatch& prefix) {
    this->cache_hits += prefix.blockIds.size();
}

void GPU::acquire(std::size_t slot) {
    this->memory[slot]->acquire();
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    this->memory[blockId]->writeToBlock(offset, value);
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return this->memory[blockId]->readFromBlock(offset);
}

void GPU::release(std::size_t slot) {
    IdleAction action = this->memory[slot]->release();

    if (action == IdleAction::Destroy || 
        (this->memory[slot]->getRefCount() == 0 && this->prefixes[slot] == std::pair<std::vector<std::string>, std::string>())) {
        delete this->memory[slot];
        this->memory[slot] = nullptr;
        this->prefixes[slot] = std::pair<std::vector<std::string>, std::string>();
    }
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    for (auto blockCopy : blocks) {
        std::pair<std::vector<std::string>, std::string> prefix = other.prefixes[blockCopy];

        std::size_t slot = 0, lowestNull = this->blocksNum + 1, lowestInactive = this->blocksNum + 1;
        bool slotFound = false;
        for (; slot < this->blocksNum; ++slot) {
            if (prefix == this->prefixes[slot]) {   
                slotFound = true;
                break;        
            } 
            
            if (this->memory[slot] == nullptr && lowestNull == this->blocksNum + 1) {
                lowestNull = slot;
            } else if (this->memory[slot] != nullptr && 
                    this->memory[slot]->getRefCount() == 0 && 
                    lowestInactive == this->blocksNum + 1) {
                // defends against inserts
                bool updateLowest = true;
                for (auto sourceSlot : blocks) {
                    if (this->prefixes[slot] == other.prefixes[sourceSlot]) {
                        updateLowest = false;
                        break;
                    }
                }
                
                if (updateLowest) lowestInactive = slot;
            }
        }

        if (slotFound) {
            continue;
        } else {
            MemoryBlock* copy = other.memory[blockCopy]->clone();
            std::size_t lowestInd;
            if (lowestNull != this->blocksNum + 1) {
                lowestInd = lowestNull;
            } else {
                lowestInd = lowestInactive;
                delete this->memory[lowestInactive];
                this->evictions++;
            }

            this->prefixes[lowestInd] = prefix;
            this->memory[lowestInd] = copy;
        }
    }
}

std::size_t GPU::getCacheHits() const {
    return this->cache_hits;
}

std::size_t GPU::getEvictions() const {
    return this->evictions;
}