#include "AntColonyOptimization.h"

std::map<double, std::pair<int, double>> AntColonyOptimization::updateSplitProbabilities(std::map<double, std::pair<int, double>> splitData)
{
    std::map<double, std::pair<int, double>> updatedSplitData; // takes into account information gain and pheremones
    for (auto split: splitData)
    {
        double infoGain = split.first;
        int attribute = split.second.first;
        double value = split.second.second;
        double pheremone = pheremoneValues[attribute][value];
        double probability = pow(infoGain, alpha) * pow(pheremone, beta);
        
        updatedSplitData[probability] = split.second;
    }
    return updatedSplitData;
}

void AntColonyOptimization::probabilisticGreedy(int node, int level, Solution& curSolution, int seed)
{
//    std::cout << "node: " << node << ", misclassified:" << curSolution.misclassifiedSamples() << std::endl;
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth || curSolution.tree[node].maxSameClass == curSolution.tree[node].nbSamplesNode)
        return;
    
    std::map<double, std::pair<int, double>> splitData;
    splitData = curSolution.getSplitData(node);
    if (splitData.size() == 0) return;
    splitData = updateSplitProbabilities(splitData);
    std::set<std::pair<int, double>> splits = chooseSplitsProbabilistically(splitData, 1, seed);
    
    auto it = splits.begin();
    /* APPLY THE SPLIT AND RECURSIVE CALL */
    int attribute = it->first;
    double threshold = it->second;
//    std::cout << "node: " << node << ", split att: " << attribute << ", split thres: " << threshold << std::endl;
    curSolution.splitOnParams(node, attribute, threshold);
    
    probabilisticGreedy(2*node+1,level+1,curSolution,seed); // Recursive call
    probabilisticGreedy(2*node+2,level+1,curSolution,seed); // Recursive call
}

void AntColonyOptimization::kStepLookAhead(int node, int level, Solution curSolution, int k, int seed)
{
//    std::cout << "node: " << node << ", misclassified: " << bestMisclassified << std::endl;
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth || curSolution.tree[node].maxSameClass == curSolution.tree[node].nbSamplesNode)
        return;
    
    std::map<double, std::pair<int, double>> splitData;
    splitData = curSolution.getSplitData(node);
    if (splitData.size() == 0) return;
    splitData = updateSplitProbabilities(splitData);
    
    const int totalSplits = 20; //pow(100000, ((double)1/k));
    std::set<std::pair<int, double>> splits = chooseSplitsProbabilistically(splitData, totalSplits, seed);
    double bestSplitAtt, bestSplitValue;
    
    /* APPLY THE SPLIT AND RECURSIVE CALL */
    for (auto split: splits)
    {
        int attribute = split.first;
        double threshold = split.second;
        curSolution.splitOnParams(node, attribute, threshold);
        
        if (level+1 < params->maxDepth &&
            curSolution.tree[node].maxSameClass != curSolution.tree[node].nbSamplesNode)
        {
            curSolution.splitSolutionKTimesBestGain(2*node+1, level+1, 0, k);
            curSolution.splitSolutionKTimesBestGain(2*node+2, level+1, 0, k);
            
            int curMisclassified = curSolution.misclassifiedSamples();
            if (curMisclassified <= bestMisclassified)
            {
                bestSolution = curSolution;
                bestMisclassified = curMisclassified;
                bestSplitAtt = attribute;
                bestSplitValue = threshold;
            }
        }
    }
//    std::cout << "att: " << bestSplitAtt << ", value: " << bestSplitValue << std::endl;
    kStepLookAhead(2*node+1,level+1,bestSolution,k,seed); // Recursive call
    kStepLookAhead(2*node+2,level+1,bestSolution,k,seed); // Recursive call
}

void AntColonyOptimization::updatePheremoneTrail(Solution solution)
{
    for (Node node: solution.tree)
    {
        int attribute = node.splitAttribute;
        double value = node.splitValue;
        if (attribute != -1)
        {
            std::map<double, int> attributeValues = pheremoneValues[attribute];
            attributeValues[value]++;
            pheremoneValues[attribute] = attributeValues;
        }
    }
}

void AntColonyOptimization::run()
{
    alpha = 2;
    beta = 1;
    
    ACO(10, 200, 1, 0); // greedy
    ACO(10, 10, 2, 3); // 3 step ACO
}



void AntColonyOptimization::ACO(int numberIterations, int numberSolutions, int type, int k)
{
    clock_t fiveMin = 300 * CLOCKS_PER_SEC;
    clock_t startTime = clock();
    Solution overallBestSolution(params);
    int overallBestMisclassified = overallBestSolution.misclassifiedSamples();
    
    int seed = 0;
    for (int it = 0; it < numberIterations; it++)
    {
        Solution curBestSolution(params);
        int curBestMisclassified = curBestSolution.misclassifiedSamples();
        for (int i = 0; i < numberSolutions; i++)
        {
            clock_t curTime = clock();
            clock_t timePassed = curTime - startTime;
            if (timePassed > fiveMin)
                break;
            
            seed++;
            Solution solution(params);
            
            switch (type) {
                case 1:
                    probabilisticGreedy(0, 0, solution, seed);
                    break;
                case 2:
                    kStepLookAhead(0, 0, solution, k, seed);
                    solution = bestSolution;
                default:
                    break;
            }
            
            int misclassified = solution.misclassifiedSamples();
            if (misclassified < curBestMisclassified)
            {
                curBestMisclassified = misclassified;
                curBestSolution = solution;
            }
        }
        updatePheremoneTrail(curBestSolution);
        if (curBestMisclassified < overallBestMisclassified)
        {
            overallBestMisclassified = curBestMisclassified;
            overallBestSolution = curBestSolution;
        }
        //        std::cout << std::endl << std::endl;
    }
    clock_t endTime = clock();
    overallBestSolution.printAndExport("oi");
    std::cout << "ACO " << type << " time: " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "(s)" << std::endl;
}
