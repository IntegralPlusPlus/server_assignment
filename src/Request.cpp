#include "Request.h"
#include "GPU.h"
#include "Model.h"

Request::Request(int id, const Model &model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
      memoryBlocks(std::vector<std::size_t>()), reservedBlocks(0)
{
}

int Request::getId() const { return this->id; }

const Model &Request::getModel() const { return this->model; }

int Request::getAdmissionCycle() const { return this->admissionCycle; }

void Request::deallocateMemory(GPU &gpu)
{
    std::size_t unreserveBlocks = this->getNeedToUnreserve();

    gpu.unreserve(unreserveBlocks, *this);

    for (auto slot : this->memoryBlocks)
    {
        gpu.release(slot);
    }
}