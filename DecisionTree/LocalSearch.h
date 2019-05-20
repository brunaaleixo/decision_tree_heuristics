#ifndef LocalSearch_h
#define LocalSearch_h

#include "Params.h"
#include "Solution.h"
#include "Utils.h"
#include "ConstructiveHeuristics.h"
#include <map>
#include <algorithm>
#include <queue>
#include <math.h>
#include <vector>
#include <string>

class LocalSearch
{
private:
    
    Params * params;         // Access to the problem and dataset parameters
    Solution bestSolution;     // Access to the solution structure to be filled
    int bestMisclassified;
    
    void propagateValueChange(int node, int level, Solution& curSolution);
    Solution fixedAttributesChangeSplitValues(int node, int level, Solution curSolution);
    
public:
    
    // Run the algorithm
    void run(Solution solution, std::string constructionMethod);
    
    // Constructor
    LocalSearch(Params * params):
    params(params), bestSolution(params) {
    };
};

#endif /* LocalSearch_h */
