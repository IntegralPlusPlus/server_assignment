#include "Request.h"
#include "Model.h"
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

