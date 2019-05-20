#include "Utils.h"

std::set<std::pair<int, double>> chooseSplitsProbabilistically(std::map<double, std::pair<int, double>>& splitData, int numberSplits, int seed = 0)
{
    srand(seed);
    std::vector<std::pair<double, double>> cumulativeInformationGains;
    double totalInformationGains = 0;
    
    for (auto it = splitData.begin(); it != splitData.end(); it++)
    {
        totalInformationGains += it->first;
        cumulativeInformationGains.push_back(std::make_pair(totalInformationGains, it->first));
    }
    double multiplicativeFactor = 100 / totalInformationGains; // we want the information gain to be 100
    totalInformationGains *= multiplicativeFactor;
    for (int i = 0; i < cumulativeInformationGains.size(); i++)
    {
        cumulativeInformationGains[i].first *= multiplicativeFactor;
    }
    std::set<std::pair<int, double>> splits;
    
    int i = 0;
    int maxIterations = 1000; // when a given split had a very small information gain, the while would
    // take really long
    while (splits.size() < std::min((int)splitData.size(), numberSplits) && i < maxIterations)
    {
        i++;
        std::pair<double, double> choice = std::make_pair((rand() % (int)(totalInformationGains)), 0);
        auto pos = std::lower_bound(cumulativeInformationGains.begin(), cumulativeInformationGains.end(), choice) - cumulativeInformationGains.begin();
        double infoGain = cumulativeInformationGains[pos].second;
        splits.insert(splitData[infoGain]);
    }
    
    return splits;
}


void keepBestSolutions(std::vector<Solution>& curSolutions, int N)
{
    std::sort(curSolutions.begin(), curSolutions.end(),
              [](Solution a, Solution b) {
                  return a.misclassifiedSamples() < b.misclassifiedSamples();
              });
    if (curSolutions.size() > N)
    {
        std::vector<int> test;
        for (Solution s: curSolutions)
        {
            test.push_back(s.misclassifiedSamples());
        }
        
        curSolutions.erase(curSolutions.begin() + N, curSolutions.end());
    }
}

std::set<double> chooseAttributeValuesProbabilistically(std::map<double, int>& attributes, int numberAttributes)
{
    int totalAttributes = std::min((int)attributes.size(), numberAttributes);
    std::set<double> selectedValues;
    
    int totalWeight = 0;
    std::vector<int> cumulativeWeights;
    std::vector<double> attributeValues;
    for (auto valueData: attributes)
    {
        attributeValues.push_back(valueData.first);
        totalWeight += valueData.second;
        cumulativeWeights.push_back(totalWeight);
    }
    while (selectedValues.size() < totalAttributes)
    {
        int choice = rand() % (totalWeight+1);
        auto pos = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), choice) - cumulativeWeights.begin();
        double value = attributeValues[pos];
        selectedValues.insert(value);
    }
    return selectedValues;
}
