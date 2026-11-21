#include "json.hpp"
#include <fstream>
#include <stdexcept>

#include "Model.h"
#include "SchedulingPolicy.h"
#include "Server.h"

using namespace std;
using json = nlohmann::json;

Server::Server(std::string configPath) : policy(NULL), currentCycle(0)
{
    Config config = readConfig(configPath);

    for (const ModelConfig modelConfig : config.models)
    {
        models.push_back(new Model(modelConfig.id, modelConfig.embeddingSize));
    }

    gpus.reserve(config.gpus.size());
    for (const GPUConfig gpuConfig : config.gpus)
    {
        gpus.push_back(GPU(gpuConfig.memoryBytes / config.blockSize, config.blockSize));
    }

    for (const RequestConfig r : config.requests)
    {
        Model *model = NULL;

        for (Model *model_ : models)
        {
            if (model_->getId() == r.modelId)
            {
                model = model_;
                break;
            }
        }
        if (model == NULL)
        {
            /* I assume this can't be true but just in case */
            continue;
        }
        switch (r.type)
        {
        case RequestType::Generative:
            nonSubmittedRequests.push_back(new GenerativeRequest(
                r.id, *model, r.admissionCycle, r.prompt, r.scriptedOutput, r.maxOutputTokens));
            break;
        case RequestType::TextEmbedder:
            nonSubmittedRequests.push_back(
                new EmbeddingRequest(r.id, *model, r.admissionCycle, r.prompt));
            break;
        }
    }

    if (config.policy == "arrivalOrder")
    {
        policy = new ArrivalOrderPolicy();
    }
    else if (config.policy == "shortestJob")
    {
        policy = new ShortestJobPolicy();
    }
    else if (config.policy == "longestPrefix")
    {
        policy = new LongestPrefixPolicy();
    }
}

bool Server::step()
{
    /*
        == Step 1. Arrival ==
        will move requests with admissionCycle = currCycle
        from nonSubmittedRequests;
        into waitingRequests;
    */

    for (size_t i = 0; i < nonSubmittedRequests.size();)
    {
        Request *request = nonSubmittedRequests[i];

        if (request->getAdmissionCycle() == currentCycle)
        {
            waitingRequests.push_back(request);
            nonSubmittedRequests.erase(nonSubmittedRequests.begin() + i);
        }
        else
        {
            ++i;
        }
    }

    /*
        == Step 2. Scheduling ==
        choose()
    */

    std::vector<const Request *> temp(waitingRequests.begin(), waitingRequests.end());

    std::vector<int> order = policy->choose(temp, gpus);
}

int Server::getCurrentCycle() { return this->currentCycle; }
Config Server::readConfig(string configPath) const
{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for (auto &gpu : config["gpus"])
    {
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for (auto &model : config["models"])
    {
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for (auto &request : config["requests"])
    {
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if (typeStr != "generative" && typeStr != "text_embedder")
        {
            throw invalid_argument("invalid request type");
        }
        RequestType type =
            typeStr == "generative" ? RequestType::Generative : RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0)
        {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0)
        {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id,     type,           modelId,        admissionCycle,
                                prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0)
    {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}
