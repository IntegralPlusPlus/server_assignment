#include <algorithm>
#include <vector>
#include <utility>
#include "SchedulingPolicy.h"
#include "GPU.h"
#include "Request.h"
#include "Model.h"

/* Sort an array of indices
   by eligble items admissions cycles */

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request *> &
                                                eligible,
                                            const std::vector<GPU> &gpus) const
{

    std::vector<int> indices(eligible.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&eligible](int a, int b)
              {
                      if (eligible[a]->getAdmissionCycle() == eligible[b]->getAdmissionCycle())
                      {
                          return eligible[a]->getId() <  eligible[b]->getId();
                      }
                      return eligible[a]->getAdmissionCycle() < eligible[b]->getAdmissionCycle(); });
    return indices;
}

SchedulingPolicy *ArrivalOrderPolicy::clone() const
{
    return new ArrivalOrderPolicy(*this);
}

/* Sort an array of indices
   by eligble items getMaxNeededMemory() */

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request *> &
                                               eligible,
                                           const std::vector<GPU> &gpus) const
{
    std::vector<int> indices(eligible.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&eligible, &gpus](int a, int b)
              {

                    std::size_t blocksA =
                        gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory());

                    std::size_t blocksB =
                        gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory());


                      if (blocksA == blocksB)
                      {
                          return eligible[a]->getId() <  eligible[b]->getId();
                      }
                      return blocksA < blocksB; });
    return indices;
}

SchedulingPolicy *ShortestJobPolicy::clone() const
{
    return new ShortestJobPolicy(*this);
}

/*
    By longest prefix
*/
std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request *> &
                                                 eligible,
                                             const std::vector<GPU> &gpus) const
{

    std::vector<std::pair<int, size_t>> indicesAndMatches(eligible.size());

    for (size_t i = 0; i < eligible.size(); ++i)
    {
        /*
            the pair is (index, longest prefix)
        */
        indicesAndMatches[i] = std::make_pair(i, 0);
        for (size_t j = 0; j < gpus.size(); ++j)
        {
            const PrefixMatch match = gpus[j].findLongestCachedPrefix(
                eligible[i]->getModel().getModelId(),
                eligible[i]->getModel().getEmbeddingSize(),
                eligible[i]->getCacheableTokens());
            indicesAndMatches[i].second = std::max(indicesAndMatches[i].second, match.matchedTokens);
        }
    }

    std::sort(indicesAndMatches.begin(),
              indicesAndMatches.end(),
              [&eligible](std::pair<int, size_t> a, std::pair<int, size_t> b)
              {
            if (a.second == b.second)
            {
                return eligible[a.first]->getId() < eligible[b.first]->getId();
            }
            return a.second > b.second; });

    std::vector<int> indices(indicesAndMatches.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = indicesAndMatches[i].first;
    }
    return indices;
}

SchedulingPolicy *LongestPrefixPolicy::clone() const
{
    return new LongestPrefixPolicy(*this);
}