#include "GPU.h"
#include "Model.h"
#include "Request.h"
#include <vector>

EmbeddingRequest::EmbeddingRequest(int id, const Model &model, int admissionCycle,
                                   std::string prompt)
    : Request(id, model, admissionCycle, prompt), tokensProcessed(0)
{
}

void EmbeddingRequest::reserveMemory(GPU &gpu)
{
    /*
        1. calculate how many blocks
        2. reserve with gpu.reserve()
        3. allocate with gpu.allocateBlock()
        4. acquire with gpu.acquire()
        5. store slot index in memoryBlocks
    */

    std::size_t blocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    gpu.reserve(blocks, *this);
    reservedBlocks = blocks;

    for (std::size_t i = 0; i < blocks; ++i)
    {
        std::size_t slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);

        --reservedBlocks;
    }
}

void EmbeddingRequest::act(GPU &gpu)
{
    /* processes one token per cycle */
    if (this->finished())
        return;
    std::size_t e = model.getEmbeddingSize();
    std::size_t slot;
    std::size_t offset;

    if (memoryBlocks.size() == 1)
    {
        slot = memoryBlocks[0];
        offset = e;
    }
    else
    {
        slot = memoryBlocks[1];
        offset = 0;
    }

    model.generateTokenEmbedding({tokens[tokensProcessed]}, gpu, slot, offset);

    for (std::size_t i = 0; i < e; ++i)
    {
        int sum = gpu.readFromBlock(memoryBlocks[0], i);
        int cur = gpu.readFromBlock(slot, offset + i);

        gpu.writeToBlock(memoryBlocks[0], i, sum + cur);
    }

    ++tokensProcessed;
    if (this->finished() && !tokens.empty())
    {
        for (std::size_t i = 0; i < e; ++i)
        {
            int sum = gpu.readFromBlock(memoryBlocks[0], i);
            this->result.push_back(sum / static_cast<int>(tokens.size()));
        }
    }
}

std::string EmbeddingRequest::getOutput() const
{
    std::string res = "";
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        if (i > 0)
        {
            res += ", ";
        }

        res += std::to_string(result[i]);
    }
    return "[" + res + "]";
}
std::size_t EmbeddingRequest::getNeedToUnreserve() const { return this->reservedBlocks; }

Request *EmbeddingRequest::clone(const Model &newModel) const
{
    EmbeddingRequest *copy = new EmbeddingRequest(id, newModel, admissionCycle, "");

    copy->tokens = tokens;
    copy->memoryBlocks = memoryBlocks;
    copy->reservedBlocks = reservedBlocks;
    copy->tokensProcessed = tokensProcessed;
    copy->result = result;

    return copy;
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const { return 2 * model.getEmbeddingSize(); }

std::size_t EmbeddingRequest::getNeededMemory(const GPU &gpu) const
{
    return 2 * model.getEmbeddingSize();
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const { return {}; }

bool EmbeddingRequest::finished() const { return tokensProcessed >= tokens.size(); }

void EmbeddingRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}