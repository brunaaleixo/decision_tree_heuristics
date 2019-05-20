#ifndef AntColonyOptimization_h
#define AntColonyOptimization_h

#include "Params.h"
#include "Solution.h"
#include "Utils.h"
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>
#include <set>

class AntColonyOptimization
{
public:
    Params * params;         // Access to the problem and dataset parameters
    double alpha, beta;
    std::map<int, std::map<double, int>> pheremoneValues;
    Solution bestSolution;
    int bestMisclassified;
    
    void updatePheremoneTrail(Solution solution);
    std::map<double, std::pair<int, double>> updateSplitProbabilities(std::map<double, std::pair<int, double>> splitData);
    void probabilisticGreedy(int node, int level, Solution& curSolution, int seed);
    void kStepLookAhead(int node, int level, Solution curSolution, int k, int seed);
    void ACO(int numberIterations, int numberSolutions, int type, int k);
    void run();
    
    AntColonyOptimization(Params * params):
    params(params), bestSolution(params)
    {
        std::map<int, std::set<double>> attributesSplitValues = params->getAttributeValues();
        for (auto attributeSplitValues: attributesSplitValues)
        {
            int attribute = attributeSplitValues.first;
            std::set<double> splitValues = attributeSplitValues.second;
            pheremoneValues[attribute] = std::map<double,int>();
            for (double value: splitValues)
            {
                pheremoneValues[attribute][value] = 1;
            }
        }
        bestMisclassified = bestSolution.misclassifiedSamples();
    };
};

#endif /* AntColonyOptimization_h */
