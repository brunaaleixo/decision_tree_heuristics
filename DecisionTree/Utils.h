#ifndef Utils_h
#define Utils_h

#include "Params.h"
#include "Solution.h"
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>

std::set<std::pair<int, double>> chooseSplitsProbabilistically(std::map<double, std::pair<int, double>>& splitData, int numberSplits, int seed);
std::set<double> chooseAttributeValuesProbabilistically(std::map<double, int>& attributeValues, int numberAttributes);

void keepBestSolutions(std::vector<Solution>& curSolutions, int N);

#endif /* Utils_hpp */
