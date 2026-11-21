#include "Request.h"
#include "Model.h"

std::size_t EmbeddingRequest::getMaxNeededMemory() const
{
    return 2 * model.getEmbeddingSize();
}