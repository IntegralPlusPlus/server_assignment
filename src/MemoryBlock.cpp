#include <cstdint>
#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size)
    : size(size),
      refCount(0),
      data(new int[size]())
{
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : size(other.size),
      refCount(0),
      data(new int[other.size]) {
    for (size_t i = 0; i < this->size; ++i) {
        data[i] = other.readFromBlock(i);
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this == &other) { // a = a
        return *this;
    }

    this->size = other.size;
    this->refCount = 0;
    delete[] data;
    this->data = new int[size];

    for (size_t i = 0; i < this->size; ++i) {
        data[i] = other.readFromBlock(i);
    }

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other)
    : size(other.size),
      refCount(0),
      data(other.data) {
    other.data = nullptr;
    other.size = 0;
    other.refCount = 0;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    this->size = other.size;
    this->refCount = 0;
    delete[] data;
    this->data = other.data;

    other.data = nullptr;
    other.size = 0;
    other.refCount = 0;

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

size_t MemoryBlock::getSize() const {
    return this->size;
}

int MemoryBlock::getRefCount() const {
    return this->refCount;
}

void MemoryBlock::setRefCount(int refCount) {
    this->refCount = refCount;
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