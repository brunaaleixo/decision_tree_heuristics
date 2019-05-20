#include "ConstructiveHeuristics.h"
#include "Utils.h"
#include <algorithm>

void ConstructiveHeuristics::run()
{
    LocalSearch localSearch(params);
    
    // Call the recursive procedure on the root node at level 0
    std::cout << "Greedy" << std::endl;
    Solution greedySol(params);
    clock_t startTime = clock();
    greedy(0, 0, greedySol);
    greedySol.printAndExport("oi");
    clock_t endTime = clock();
    std::cout << "----- Greedy time " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "(s)" << std::endl;
    localSearch.run(greedySol, "greedy");
    
    for (int k = 1; k < params->maxDepth; k++)
    {
        std::cout << k << " step Look ahead" << std::endl;
        Solution lookAheadSol(params);
        bestSolution = lookAheadSol;
        bestMisclassified = bestSolution.misclassifiedSamples();
        
        startTime = clock();
        kStepLookAhead(0, 0, lookAheadSol, k);
        endTime = clock();
        
        bestSolution = pruneTree(0, 0, bestSolution);
        bestSolution.printAndExport("oi");
        std::cout << k << "step time " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "(s)" << std::endl;
        
        std::string constructionHeuristic = std::to_string(k) + " step look ahead";
        localSearch.run(bestSolution, constructionHeuristic);
    }
    std::cout << "Racing" << std::endl;
    Solution racingSol(params);
    bestSolution = racingSol;
    bestMisclassified = bestSolution.misclassifiedSamples();
    std::vector<Solution> solutions;
    solutions.push_back(racingSol);
    
    startTime = clock();
    racing(0, 0, solutions);
    endTime = clock();
    
    bestSolution = pruneTree(0, 0, bestSolution);
    bestSolution.printAndExport("oi");
    localSearch.run(bestSolution, "racing");
    std::cout << "racing time " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "(s)" << std::endl;
    
}

/*
 Greedy
 */
void ConstructiveHeuristics::greedy(int node, int level, Solution& curSolution)
{
//    std::cout << "node: " << node << ", misclassified:" << curSolution.misclassifiedSamples() << std::endl;
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth || curSolution.tree[node].maxSameClass == curSolution.tree[node].nbSamplesNode)
    {
        return;
    }
    
    std::map<double, std::pair<int, double>> splitData;
    splitData = curSolution.getSplitData(node);
    if (splitData.size() == 0) return;
    
    auto it = splitData.rbegin();
    /* APPLY THE SPLIT AND RECURSIVE CALL */
    int attribute = it->second.first;
    double threshold = it->second.second;
    
    curSolution.splitOnParams(node, attribute, threshold);
    
    greedy(2*node+1,level+1,curSolution); // Recursive call
    greedy(2*node+2,level+1,curSolution); // Recursive call
}

/*
 Look ahead
 */
void ConstructiveHeuristics::kStepLookAhead(int node, int level, Solution curSolution, int k)
{
//    std::cout << "node: " << node << ", misclassified: " << bestMisclassified << std::endl;
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth || curSolution.tree[node].maxSameClass == curSolution.tree[node].nbSamplesNode)
        return;
    
    std::map<double, std::pair<int, double>> splitData;
    splitData = curSolution.getSplitData(node);

    if (splitData.size() == 0) return;
    const int totalSplits = 400/k;
    std::set<std::pair<int, double>> splits = chooseSplitsProbabilistically(splitData, totalSplits, 1);
    
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
            }
        }
    }
    
    kStepLookAhead(2*node+1,level+1,bestSolution,k); // Recursive call
    kStepLookAhead(2*node+2,level+1,bestSolution,k); // Recursive call
}

/*
 Racing
 */
std::vector<Solution> ConstructiveHeuristics::racing(int node, int level, std::vector<Solution> curSolutions)
{
    const int totalSolutions = 100;
    const int totalSplits = 10;
    int allSolutionsFinished = 1;
    
    std::vector<Solution> updatedSolutions;
    /* Try different splits for each of the current solutions */
    if (level < params->maxDepth)
    {
        for (Solution solution: curSolutions)
        {
            /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
            if (solution.tree[node].maxSameClass == solution.tree[node].nbSamplesNode)
            {
                updatedSolutions.push_back(solution);
                continue;
            }
            
            std::map<double, std::pair<int, double>> splitData;
            splitData = solution.getSplitData(node);
            if (splitData.size() == 0) // contradictions in the data - no possible improving sets
                continue;
            
            allSolutionsFinished = 0;
            std::set<std::pair<int, double>> splits = chooseSplitsProbabilistically(splitData, totalSplits, 0);
            for (auto split: splits)
            {
                int attribute = split.first;
                double threshold = split.second;
                solution.splitOnParams(node, attribute, threshold);
                updatedSolutions.push_back(solution);
            }
        }
        
        keepBestSolutions(updatedSolutions, totalSolutions);

        if (!allSolutionsFinished)
        {
            updatedSolutions = racing(2*node+1,level+1,updatedSolutions); // Recursive call
            updatedSolutions = racing(2*node+2,level+1,updatedSolutions); // Recursive call
        }
    }
    if (allSolutionsFinished)
    {
        int curMisclassified = curSolutions[0].misclassifiedSamples();
        if (curMisclassified < bestMisclassified)
        {
            bestSolution = curSolutions[0];
        }
        return curSolutions;
    }
    return updatedSolutions;
}

/*
 *  Pruning
 */
Solution ConstructiveHeuristics::pruneTree(int node, int level, Solution solution)
{
    Node nodeData = solution.tree[node];
    if (level == params->maxDepth || nodeData.nodeType == Node::NODE_LEAF)
        return solution;
    
    Node leftNodeData = solution.tree[2*node+1];
    Node rightNodeData = solution.tree[2*node+2];
    if (nodeData.nodeType == Node::NODE_INTERNAL &&
        leftNodeData.nodeType == Node::NODE_LEAF && rightNodeData.nodeType == Node::NODE_LEAF)
    {
        int misclass = nodeData.nodeMisclassifications();
        int leftMisclass = leftNodeData.nodeMisclassifications();
        int rightMisclass = rightNodeData.nodeMisclassifications();
        // split was useless - set node as leaf
        if (leftMisclass + rightMisclass >= misclass)
        {
            solution.pruneNode(node, level);
            return solution;
        }
    }

    solution = pruneTree(2*node+1, level+1, solution);
    solution = pruneTree(2*node+2, level+1, solution);
    return solution;
}
