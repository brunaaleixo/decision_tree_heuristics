#include "LocalSearch.h"

void LocalSearch::run(Solution solution, std::string constructionMethod)
{
    clock_t startTime = clock();
    bestSolution = solution;
    bestMisclassified = solution.misclassifiedSamples();
    fixedAttributesChangeSplitValues(0, 0, solution);
    clock_t endTime = clock();
    bestSolution.printAndExport(constructionMethod + "LocalSearch");
    std::cout << constructionMethod << (endTime - startTime) / (double)CLOCKS_PER_SEC << "(s)" << std::endl;
}

/*
 If a split value is modified, the number of samples in each of its descendants
 may consequently change as well. Therefore, the whole tree should be reevaluated
 */
void LocalSearch::propagateValueChange(int node, int level, Solution& curSolution)
{
    Node nodeData = curSolution.tree[node];
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth ||
        nodeData.maxSameClass == nodeData.nbSamplesNode)
        return;
    if (nodeData.splitAttribute != -1)
    {
        curSolution.splitOnParams(node, nodeData.splitAttribute, nodeData.splitValue);
        
        propagateValueChange(2*node+1,level+1,curSolution); // Recursive call
        propagateValueChange(2*node+2,level+1,curSolution); // Recursive call
    }
}

Solution LocalSearch::fixedAttributesChangeSplitValues(int node, int level, Solution curSolution)
{
    /* BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS */
    if (level >= params->maxDepth || curSolution.tree[node].maxSameClass == curSolution.tree[node].nbSamplesNode)
        return curSolution;
    
    int attribute = curSolution.tree[node].splitAttribute;

    const int nAttributes = 10;
    std::set<double> values = chooseAttributeValuesProbabilistically(curSolution.bestSplitsForAttributes[attribute], nAttributes);
    
    for (auto splitValue: values)
    {
        int threshold = splitValue;
        curSolution.splitOnParams(node, attribute, threshold);
        propagateValueChange(node, level, curSolution);
        int curMisclassified = curSolution.misclassifiedSamples();
        if (curMisclassified <= bestMisclassified)
        {
            bestSolution = curSolution;
            bestMisclassified = curMisclassified;
        }
        curSolution = fixedAttributesChangeSplitValues(2*node+1, level+1, curSolution);
        curSolution = fixedAttributesChangeSplitValues(2*node+2, level+1, curSolution);
    }
    return bestSolution;
}
