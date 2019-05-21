#ifndef SOLUTION_H
#define SOLUTION_H

#include <vector>
#include <queue>
#include <map>
#include "Params.h"
#include <unordered_map>

// Structure representing one node of the (orthogonal) decision tree or a leaf
class Node
{

public:

	enum {NODE_NULL, NODE_LEAF, NODE_INTERNAL} nodeType;	// Node type
	Params * params;										// Access to the problem and dataset parameters
	int splitAttribute;										// Attribute to which the split is applied (filled through the greedy algorithm)
	double splitValue;											// Threshold value for the split (for numerical attributes the left branch will be <= splitValue, for categorical will be == splitValue)					
	std::vector <int> samples;								// Samples from the training set at this node
	std::vector <int> nbSamplesClass;						// Number of samples of each class at this node (for each class)
	int nbSamplesNode;										// Total number of samples in this node
	int majorityClass;										// Majority class in this node
	int maxSameClass;										// Maximum number of elements of the same class in this node
	double entropy;											// Entropy in this node
	
	void evaluate()
	{	
		entropy = 0.0;
		for (int c = 0; c < params->nbClasses; c++)
		{
			if (nbSamplesClass[c] > 0)
			{
				double frac = (double)nbSamplesClass[c]/(double)nbSamplesNode;
				entropy -= frac * log2(frac);
				if (nbSamplesClass[c] > maxSameClass)
				{ 
					maxSameClass = nbSamplesClass[c];
					majorityClass = c;
				}
			}
		}
	}

    int nodeMisclassifications()
    {
        return nbSamplesNode - nbSamplesClass[majorityClass];
    }
    
    void resetSamples()
    {
        maxSameClass = 0;
        majorityClass = 0;
        samples.clear();
        std::fill(nbSamplesClass.begin(), nbSamplesClass.end(), 0);
        nbSamplesNode = 0;
    }
    
	void addSample(int i)
	{
		samples.push_back(i);
		nbSamplesClass[params->dataClasses[i]]++;
		nbSamplesNode++;
	}
    
    void emptyNode()
    {
        nodeType = NODE_NULL;
        splitAttribute = -1;
        splitValue = -1.e30;
        nbSamplesClass = std::vector<int>(params->nbClasses, 0);
        nbSamplesNode = 0;
        majorityClass = -1;
        maxSameClass = 0;
        entropy = -1.e30;
        resetSamples();
    }

	Node(Params * params):params(params)
	{
        emptyNode();
	}
};

class Solution
{

private:

	// Access to the problem and dataset parameters
	Params * params;
   
public:

	// Vector representing the tree
	// Parent of tree[k]: tree[(k-1)/2]
	// Left child of tree[k]: tree[2*k+1]
	// Right child of tree[k]: tree[2*k+2]
	std::vector <Node> tree;
    
    std::map<int, std::map<double, int>> bestSplitsForAttributes;
    
    double getNumericalInfoGain(int node, int indexSample, std::vector<int>& nbSamplesClassLeft, std::vector<int>& nbSamplesClassRight)
    {
        int nbSamplesNode = tree[node].nbSamplesNode;
        double originalEntropy = tree[node].entropy;
        
        // Evaluate entropy of the two resulting sample sets
        double entropyLeft = 0.0;
        double entropyRight = 0.0;
        for (int c = 0; c < params->nbClasses; c++)
        {
            // Remark that indexSample contains at this stage the number of samples in the left
            if (nbSamplesClassLeft[c] > 0)
            {
                double fracLeft = (double)nbSamplesClassLeft[c] / (double)(indexSample);
                entropyLeft -= fracLeft * log2(fracLeft);
            }
            if (nbSamplesClassRight[c] > 0)
            {
                double fracRight = (double)nbSamplesClassRight[c] / (double)(nbSamplesNode - indexSample);
                entropyRight -= fracRight * log2(fracRight);
            }
        }
        
        // Evaluate the information gain and store if this is the best option found until now
        double informationGain = originalEntropy - ((double)indexSample*entropyLeft + (double)(nbSamplesNode - indexSample)*entropyRight) / (double)nbSamplesNode;
        
        return informationGain;
    }
    
    double getCategoricalInfoGain(int node, int level, std::vector<int>& nbSamplesLevel, std::vector<int>& nbSamplesClass, std::vector<std::vector<int>>& nbSamplesLevelClass)
    {
        int nbSamplesNode = tree[node].nbSamplesNode;
        double originalEntropy = tree[node].entropy;
        
        double entropyLevel = 0.0;
        double entropyOthers = 0.0;
        for (int c = 0; c < params->nbClasses; c++)
        {
            if (nbSamplesLevelClass[level][c] > 0)
            {
                double fracLevel = (double)nbSamplesLevelClass[level][c] / (double)nbSamplesLevel[level] ;
                entropyLevel -= fracLevel * log2(fracLevel);
            }
            if (nbSamplesClass[c] - nbSamplesLevelClass[level][c] > 0)
            {
                double fracOthers = (double)(nbSamplesClass[c] - nbSamplesLevelClass[level][c]) / (double)(nbSamplesNode - nbSamplesLevel[level]);
                entropyOthers -= fracOthers * log2(fracOthers);
            }
        }
        
        // Evaluate the information gain and store if this is the best option found until now
        double informationGain = originalEntropy - ((double)nbSamplesLevel[level] *entropyLevel + (double)(nbSamplesNode - nbSamplesLevel[level])*entropyOthers) / (double)nbSamplesNode;
        return informationGain;
    }
    
    std::map<double, std::pair<int, double>> getAttributeSplits(int node, int att, std::map<double, std::pair<int, double>>& splitData)
    {
        int nbSamplesNode = tree[node].nbSamplesNode;
        double bestAttributeValue = 0;
        double bestInfoGain = 0;
        
        if (params->attributeTypes[att] == TYPE_NUMERICAL)
        {
            /* CASE 1) -- GET SPLITS FOR NUMERICAL ATTRIBUTE c */
            
            // Define some data structures
            std::vector <std::pair<double, int>> orderedSamples;        // Order of the samples according to attribute c
            std::set<double> attributeLevels;                           // Store the possible levels of this attribute among the samples (will allow to "skip" samples with equal attribute value)
            for (int s : tree[node].samples)
            {
                orderedSamples.push_back(std::pair<double, int>(params->dataAttributes[s][att], params->dataClasses[s]));
                attributeLevels.insert(params->dataAttributes[s][att]);
            }
            // If all sample have the same level for this attribute, it's useless to look for a split
            if (attributeLevels.size() > 1) {
                std::sort(orderedSamples.begin(), orderedSamples.end());
                
                // Initially all samples are on the right
                std::vector <int> nbSamplesClassLeft = std::vector<int>(params->nbClasses, 0);
                std::vector <int> nbSamplesClassRight = tree[node].nbSamplesClass;
                int indexSample = 0;
                for (double attributeValue : attributeLevels) // Go through all possible attribute values in increasing order
                {
                    // Iterate on all samples with this attributeValue and switch them to the left
                    while (indexSample < nbSamplesNode && orderedSamples[indexSample].first < attributeValue + MY_EPSILON)
                    {
                        nbSamplesClassLeft[orderedSamples[indexSample].second]++;
                        nbSamplesClassRight[orderedSamples[indexSample].second]--;
                        indexSample++;
                    }
                    
                    if (indexSample != nbSamplesNode) // No need to consider the case in which all samples have been switched to the left
                    {
                        double informationGain = getNumericalInfoGain(node, indexSample, nbSamplesClassLeft, nbSamplesClassRight);
                        std::pair<int, double> split = std::make_pair(att, attributeValue);
                        splitData.insert(std::make_pair(informationGain, split));
                        if (informationGain > bestInfoGain)
                        {
                            bestInfoGain = informationGain;
                            bestAttributeValue = attributeValue;
                        }
                    }
                }
            }
            if (bestSplitsForAttributes[att][bestAttributeValue])
            {
                bestSplitsForAttributes[att][bestAttributeValue]++;
            } else
            {
                bestSplitsForAttributes[att][bestAttributeValue] = 1;
            }
        }
        else
        {
            /* CASE 2) -- GET SPLITS FOR CATEGORICAL ATTRIBUTE c */
            
            // Count for each level of attribute c and each class the number of samples
            std::vector <int> nbSamplesLevel = std::vector <int>(params->nbLevels[att],0);
            std::vector <int> nbSamplesClass = std::vector <int>(params->nbClasses, 0);
            std::vector < std::vector <int> > nbSamplesLevelClass = std::vector< std::vector <int> >(params->nbLevels[att], std::vector <int>(params->nbClasses,0));
            for (int s : tree[node].samples)
            {
                nbSamplesLevel[params->dataAttributes[s][att]]++;
                nbSamplesClass[params->dataClasses[s]]++;
                nbSamplesLevelClass[params->dataAttributes[s][att]][params->dataClasses[s]]++;
            }
            
            // Calculate information gain for a split at each possible level of attribute c
            for (int level = 0; level < params->nbLevels[att]; level++)
            {
                if (nbSamplesLevel[level] > 0 && nbSamplesLevel[level] < nbSamplesNode)
                {
                    // Evaluate entropy of the two resulting sample sets
                    double informationGain = getCategoricalInfoGain(node, level, nbSamplesLevel, nbSamplesClass, nbSamplesLevelClass);
              
                    std::pair<int, double> split = std::make_pair(att, level);
                    splitData.insert(std::make_pair(informationGain, split));
                }
            }
        }
        return splitData;
    }
    
    // Get possible splits at node for solution
    std::map<double, std::pair<int, double>> getSplitData(int node)
    {
        std::map<double, std::pair<int, double>> splitData;
        
        for (int att = 0; att < params->nbAttributes; att++)
        {
            getAttributeSplits(node, att, splitData);
        }
        return splitData;
    }
    
    void splitSolutionKTimesBestGain(int node, int level, int curStep, int k)
    {
        if (curStep == k || level >= params->maxDepth)
            return;

        std::map<double, std::pair<int, double>> splitData;
        splitData = getSplitData(node);
        if (splitData.size() == 0) return;
        
        auto bestSplit = splitData.rbegin();
        int attribute = bestSplit->second.first;
        double threshold = bestSplit->second.second;
        splitOnParams(node, attribute, threshold);
        
        splitSolutionKTimesBestGain(2*node+1, level+1, curStep+1, k);
        splitSolutionKTimesBestGain(2*node+2, level+1, curStep+1, k);
    }
    
    void splitOnParams(int node, int attribute, double threshold)
    {
        tree[node].splitAttribute = attribute;
        tree[node].splitValue = threshold;
        tree[node].nodeType = Node::NODE_INTERNAL;
        tree[2*node+1].nodeType = Node::NODE_LEAF ;
        tree[2*node+2].nodeType = Node::NODE_LEAF ;
        tree[2*node+1].resetSamples();
        tree[2*node+2].resetSamples();
        for (int s : tree[node].samples)
        {
            if ((params->attributeTypes[attribute] == TYPE_NUMERICAL &&
                 params->dataAttributes[s][attribute] < threshold + MY_EPSILON )||
                (params->attributeTypes[attribute] == TYPE_CATEGORICAL &&
                 params->dataAttributes[s][attribute] < threshold + MY_EPSILON &&
                 params->dataAttributes[s][attribute] > threshold - MY_EPSILON))
                tree[2*node+1].addSample(s);
            else
                tree[2*node+2].addSample(s);
        }
        tree[2*node+1].evaluate(); // Setting all other data structures
        tree[2*node+2].evaluate(); // Setting all other data structures
    }
    
    void pruneNode(int node, int level)
    {
        tree[node].nodeType = Node::NODE_LEAF;
        tree[node].splitAttribute = -1;
        tree[node].splitValue = -1.e30;
        tree[2*node+1].emptyNode();
        tree[2*node+2].emptyNode();
    }
    
    // Misclassified
    int misclassifiedSamples()
    {
        int nbMisclassifiedSamples = 0;
        for (int d = 0; d <= params->maxDepth; d++)
        {
            for (int i = pow(2, d) - 1; i < pow(2, d + 1) - 1; i++)
            {
                if (tree[i].nodeType == Node::NODE_LEAF)
                {
                    int misclass = tree[i].nodeMisclassifications();
                    nbMisclassifiedSamples += misclass;
                }
            }
        }
        return nbMisclassifiedSamples;
    }
    
    // Prints the final solution
	void printAndExport(std::string fileName)
	{
		int nbMisclassifiedSamples = 0;
		std::cout << std::endl << "---------------------------------------- PRINTING SOLUTION ----------------------------------------" << std::endl;
		for (int d = 0; d <= params->maxDepth; d++)
		{
			// Printing one complete level of the tree
			for (int i = pow(2, d) - 1; i < pow(2, d + 1) - 1; i++)
			{
				if (tree[i].nodeType == Node::NODE_INTERNAL)
					std::cout << "(N" << i << ",A[" << tree[i].splitAttribute << "]" << (params->attributeTypes[tree[i].splitAttribute] == TYPE_NUMERICAL ? "<=" : "=") << tree[i].splitValue << ") ";
				else if (tree[i].nodeType == Node::NODE_LEAF)
				{
					int misclass = tree[i].nodeMisclassifications();
					nbMisclassifiedSamples += misclass;
					std::cout << "(L" << i << ",C" << tree[i].majorityClass << "," << tree[i].nbSamplesClass[tree[i].majorityClass] << "," << misclass << ") ";
				}
			}
			std::cout << std::endl;
		}
		std::cout << nbMisclassifiedSamples << "/" << params->nbSamples << " MISCLASSIFIED SAMPLES" << std::endl;
		std::cout << "---------------------------------------------------------------------------------------------------" << std::endl << std::endl;

		std::ofstream myfile;
		myfile.open(fileName.data());
		if (myfile.is_open())
		{
			myfile << "TIME(s): " << (params->endTime - params->startTime) / (double)CLOCKS_PER_SEC << std::endl;
			myfile << "NB_SAMPLES: " << params->nbSamples << std::endl;
			myfile << "NB_MISCLASSIFIED: " << nbMisclassifiedSamples << std::endl;
			myfile.close();
		}
		else
			std::cout << "----- IMPOSSIBLE TO OPEN SOLUTION FILE: " << params->pathToSolution << " ----- " << std::endl;
	}
    
	Solution(Params * params):params(params)
	{
		// Initializing tree data structure and the nodes inside -- The size of the tree is 2^{maxDepth} - 1
		tree = std::vector<Node>(pow(2,params->maxDepth+1)-1,Node(params));

		// The tree is initially made of a single leaf (the root node)
		tree[0].nodeType = Node::NODE_LEAF;
		for (int i = 0; i < params->nbSamples; i++) 
			tree[0].addSample(i);
		tree[0].evaluate();
	};
};
#endif
