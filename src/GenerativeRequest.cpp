#include "Request.h"
#include "Model.h"
#include "GPU.h"
#include <algorithm>

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, 
                                     std::string prompt, std::string scriptedOutput,
                                     std::size_t maxOutputTokens):
                                     Request(id, model, admissionCycle, prompt),
                                     scriptedOutput(scriptedOutput),
                                     maxOutputTokens(maxOutputTokens),
                                     tokensOutput(model.tokenizer(scriptedOutput)),
                                     tokensProcessed(0) {
}

bool GenerativeRequest::finished() const {
    return this->tokensProcessed == this->tokens.size() + std::min(this->maxOutputTokens, this->tokensOutput.size());
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return this->tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return this->model.getEmbeddingSize() * (this->tokens.size() + this->maxOutputTokens);
}

std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    PrefixMatch prefix = gpu.findLongestCachedPrefix(this->model.getModelId(), 
                                                     this->model.getEmbeddingSize(),
                                                     tokens);
    
    return this->model.getEmbeddingSize() * (this->tokens.size() + this->maxOutputTokens - prefix.matchedTokens);
}

