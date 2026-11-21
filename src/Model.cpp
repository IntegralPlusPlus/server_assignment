#include <cstdint>
#include "Model.h"
#include "GPU.h"

/* Constructor with initizlier list */
Model::Model(const std::string &id, std::size_t embeddingSize) : modelId(id), embeddingSize(embeddingSize)
{
}

/* Removes non-alpha-numeric characters
   and splits string into words using space as delimiter

*/
std::vector<std::string> Model::tokenizer(std::string text) const
{
    std::vector<std::string> result;
    std::string currWord = "";
    for (std::size_t i = 0; i < text.length(); ++i)
    {
        char ascii = text[i];
        /*
            [48,57] digits
            [65,90] UPPERCASE LETTERS
            [97,122] lowercase letters
            [32] -- space
        */

        if (ascii == 32)
        {
            if (!currWord.empty())
            {
                result.push_back(currWord);
                currWord = "";
            }

            continue;
        }

        bool isAlphaNumeric = (ascii >= 48 && ascii <= 57) || (ascii >= 65 && ascii <= 90) || (ascii >= 97 && ascii <= 122);
        if (!isAlphaNumeric)
        {
            continue;
        }

        currWord += ascii;
    }
    if (!currWord.empty())
    {
        result.push_back(currWord);
    }
    return result;
}

std::uint64_t Model::mixByte(std::uint64_t state, unsigned char byte) const
{
    state ^= byte;
    state *= 1099511628211ULL;
    return state;
}

std::uint64_t Model::mixString(std::uint64_t state, const std::string &text) const
{
    const std::uint64_t length = static_cast<std::uint64_t>(text.size());
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
        state = mixByte(state, static_cast<unsigned char>((length >> shift) & 0xFFULL));
    }
    for (unsigned char byte : text)
    {
        state = mixByte(state, byte);
    }
    return state;
}

// Produces reproducible synthetic vectors for the simulation.
// The constants come from FNV-1a and SplitMix64; these are not real text embeddings.
void Model::generateTokenEmbedding(const std::vector<std::string> &tokens, GPU &gpu, size_t slot,
                                   size_t offset) const
{
    // The tokens are hashed joined by single spaces (a token never contains a space).
    std::string token;
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        if (i > 0)
        {
            token += " ";
        }
        token += tokens[i];
    }
    // Length-prefix both fields before hashing their bytes with FNV-1a.
    std::uint64_t state = 14695981039346656037ULL;

    state = mixString(state, modelId);
    state = mixString(state, token);

    // Generate one deterministic value per dimension.
    for (std::size_t i = 0; i < embeddingSize; ++i)
    {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        gpu.writeToBlock(slot, offset + i, static_cast<int>(value % 2001ULL) - 1000);
    }
}

std::size_t Model::getEmbeddingSize() const
{
    return this->embeddingSize;
}

std::string Model::getId() const {
    return this->modelId;
}