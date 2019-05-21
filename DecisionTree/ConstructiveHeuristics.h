#ifndef ConstructiveHeuristics_h
#define ConstructiveHeuristics_h

#include "Params.h"
#include "Solution.h"
#include "LocalSearch.h"
#include <map>
#include <algorithm>
#include <queue>
#include <math.h>
#include <vector>
#include <string>

class ConstructiveHeuristics
{
private:
    
    Params * params;         // Access to the problem and dataset parameters
    Solution bestSolution;     // Access to the solution structure to be filled

    int didSetInitialSolution = 0;
    int bestMisclassified;
    
    double getNumericalInfoGain(int nbSamplesNode, double originalEntropy, int indexSample, std::vector<int>& nbSamplesClassLeft, std::vector<int>& nbSamplesClassRight);
    double getCategoricalInfoGain(int nbSamplesNode, double originalEntropy, int level, std::vector<int>& nbSamplesLevel, std::vector<int>& nbSamplesClass, std::vector<std::vector<int>>& nbSamplesLevelClass);
    
    void kStepLookAhead(int node, int level, Solution curSolution, int k);
    std::vector<Solution> racing(int node, int level, std::vector<Solution> curSolutions);
    void greedy(int node, int level, Solution& curSolution);
    
    Solution pruneTree(int node, int level, Solution solution);
    
    void recursiveConstruction(int node, int level, Solution& solution);
    
public:
    // Run the algorithm
    void run();
    
    // Constructor
    ConstructiveHeuristics(Params * params):
    params(params), bestSolution(params) {};
};

#endif /* ConstructiveHeuristics_h */
