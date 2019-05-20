#include "Commandline.h"
#include "Params.h"
#include "Solution.h"
#include "ConstructiveHeuristics.h"
#include "LocalSearch.h"
#include "AntColonyOptimization.h"
int main(int argc, char *argv[])
{
	Commandline c(argc, argv);
	if (c.is_valid())
	{
		// Initialization of the problem data from the commandline
		Params params(c.get_path_to_instance(), c.get_path_to_solution(), c.get_seed(), c.get_maxDepth(), c.get_cpu_time() * CLOCKS_PER_SEC);

        ConstructiveHeuristics solver(&params);
        solver.run();
		
        AntColonyOptimization aco(&params);
        aco.run();
        
		std::cout << "----- END OF ALGORITHM" << std::endl;
	}
	return 0;
}
