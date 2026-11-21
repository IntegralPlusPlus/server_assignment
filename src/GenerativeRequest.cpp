#include "GPU.h"
#include "Model.h"
#include "Request.h"
#include <algorithm>

GenerativeRequest::GenerativeRequest(int id, const Model &model, int admissionCycle,
                                     std::string prompt, std::string scriptedOutput,
                                     std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput),
      maxOutputTokens(maxOutputTokens), tokensOutput(model.tokenizer(scriptedOutput)),
      tokensProcessed(0), prefix(std::vector<std::string>())
{
}

bool GenerativeRequest::finished() const
{
    return this->tokensProcessed ==
           this->tokens.size() + std::min(this->maxOutputTokens, this->tokensOutput.size());
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const { return this->tokens; }

void GenerativeRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}

std::size_t GenerativeRequest::getMaxNeededMemory() const
{
    return this->model.getEmbeddingSize() * (this->tokens.size() + this->maxOutputTokens);
}

std::size_t GenerativeRequest::getNeededMemory(const GPU &gpu) const
{
    PrefixMatch cache =
        gpu.findLongestCachedPrefix(this->model.getId(), this->model.getEmbeddingSize(), tokens);

    return this->model.getEmbeddingSize() *
           (this->tokens.size() + this->maxOutputTokens - cache.matchedTokens);
}

void GenerativeRequest::reserveMemory(GPU &gpu)
{
    PrefixMatch cache = gpu.findLongestCachedPrefix(this->model.getId(),
                                                    this->model.getEmbeddingSize(), this->tokens);

    this->memoryBlocks = cache.blockIds;
    this->tokensProcessed += cache.matchedTokens;
    gpu.acquireCacheHits(cache);
    for (auto block : cache.blockIds)
    {
        gpu.acquire(block);
    }

    for (std::size_t i = 0; i < cache.matchedTokens; ++i)
    {
        this->prefix.push_back(this->tokens[i]);
    }

    std::size_t neccessaryMemory =
        this->model.getEmbeddingSize() *
        (this->tokens.size() + this->maxOutputTokens - cache.matchedTokens);
    this->reservedBlocks = gpu.getBlocksForMemory(neccessaryMemory);
    gpu.reserve(this->reservedBlocks, *this);
}

void GenerativeRequest::act(GPU &gpu)
{
    if (this->finished())
        return;

    std::string currToken;
    if (this->tokensProcessed < this->tokens.size())
    {
        currToken = this->tokens[this->tokensProcessed];
    }
    else
    {
        currToken = this->tokensOutput[this->tokensProcessed - this->tokens.size()];
    }

    this->prefix.push_back(currToken);

    std::size_t tokensPerBlock = gpu.getBlockSize() / this->model.getEmbeddingSize();
    std::size_t slot;

    if (this->tokensProcessed % tokensPerBlock == 0)
    {
        slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        this->reservedBlocks--;
    }
    else
    {
        slot = this->memoryBlocks[this->tokensProcessed / tokensPerBlock];
    }

    std::size_t offset = (this->tokensProcessed % tokensPerBlock) * model.getEmbeddingSize();
    this->model.generateTokenEmbedding(this->prefix, gpu, slot, offset);

    this->tokensProcessed++;
    if (this->tokensProcessed % tokensPerBlock == 0)
    {
        gpu.assignPrefixToBlock(slot, prefix, this->model.getId());
    }
}

std::string GenerativeRequest::getOutput() const
{
    std::string output;

    for (std::size_t i = this->tokens.size(); i < this->prefix.size(); ++i)
    {
        output += this->prefix[i];
        if (i != this->prefix.size() - 1)
            output += " ";
    }

    return output;
}

std::size_t GenerativeRequest::getNeedToUnreserve() const { return this->reservedBlocks; }