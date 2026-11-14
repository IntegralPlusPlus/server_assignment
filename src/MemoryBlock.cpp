#include <cstdint>
#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size) {
    this->size = size;
    this->data = new int[size]();
    this->refCount = 0;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other) {
    this->size = other.size;
    this->refCount = 0;
    delete[] this->data;

    this->data = new int[size];

    for (size_t i = 0; i < this->size; ++i) {
        data[i] = other.readFromBlock(i);
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    this->size = other.size;
    this->refCount = 0;
    delete[] this->data;

    this->data = new int[size];

    for (size_t i = 0; i < this->size; ++i) {
        data[i] = other.readFromBlock(i);
    }

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) {
    this->size = other.size;
    this->refCount = 0;
    delete[] this->data;

    this->data = other.data;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    this->size = other.size;
    this->refCount = 0;
    this->data = other.data;
    delete[] this->data;

    return *this;
}

MemoryBlock::~MemoryBlock() {
    delete[] data;
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    this->data[offset] = value;
}

int MemoryBlock::readFromBlock(std::size_t offset) const {
    return this->data[offset];
}

void MemoryBlock::acquire() {
    this->refCount++;
}

IdleAction MemoryBlock::release() {
    this->refCount--;
    if (this->refCount == 0) {
        return this->onLastRelease();
    } else {
        return IdleAction::KeepCached;
    }
}

KVBlock::KVBlock(size_t size): MemoryBlock(size) {
}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    MemoryBlock* copy = new KVBlock(this->size);

    for (std::size_t i = 0; i < this->size; ++i) {
        copy->writeToBlock(i, this->readFromBlock(i));
    }

    return copy;
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size): MemoryBlock(size) {
}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    MemoryBlock* copy = new EmbeddingWorkspace(this->size);

    for (std::size_t i = 0; i < this->size; ++i) {
        copy->writeToBlock(i, this->readFromBlock(i));
    }

    return copy;   
}