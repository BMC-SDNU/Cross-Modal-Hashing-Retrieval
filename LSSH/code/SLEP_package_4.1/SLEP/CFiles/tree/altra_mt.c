#include "mex.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "altra.h"


/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * altra solves the following problem
 *
 * 1/2 \|x-v\|^2 + \sum \lambda_i \|x_{G_i}\|,
 *
 * where x and v are of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * The file is implemented in the following in Matlab:
 *
 * x=altra(v, n, ind, nodes);
 *
 * ind is a 3 x nodes matrix.
 *       Each column corresponds to a node.
 *
 *       The first element of each column is the starting index,
 *       the second element of each column is the ending index
 *       the third element of each column corrreponds to \lambbda_i.
 *
 * -------------------------------------------------------------------
 *                       Notices:
 * -------------------------------------------------------------------
 *
 * 1. The nodes in the parameter "ind" should be given in the 
 *    either
 *           the postordering of depth-first traversal
 *    or 
 *           the reverse breadth-first traversal.
 *
 * 2. When each elements of x are penalized via the same L1 
 *    (equivalent to the L2 norm) parameter, one can simplify the input
 *    by specifying 
 *           the "first" column of ind as (-1, -1, lambda)
 *
 *    In this case, we treat it as a single "super" node. Thus in the value
 *    nodes, we only count it once.
 *
 * 3. The values in "ind" are in [1,n].
 *
 * 4. The third element of each column should be positive. The program does
 *    not check the validity of the parameter. 
 *
 *    It is still valid to use the zero regularization parameter.
 *    In this case, the program does not change the values of 
 *    correponding indices.
 *    
 *
 * -------------------------------------------------------------------
 *                       History:
 * -------------------------------------------------------------------
 *
 * Composed by Jun Liu on April 20, 2010
 *
 * For any question or suggestion, please email j.liu@asu.edu.
 *
 */


/*
 * altra_mt is a generalization of altra to the multi-task learning scenario
 *
 * altra_mt(X, V, n, k, ind, nodes);
 *
 * It applies altra for each row (1xk) of X and V
 *
 */


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    double*		V		=	mxGetPr(prhs[0]);
    int			n		=   (int) mxGetScalar(prhs[1]);
    int			k		=   (int) mxGetScalar(prhs[2]);
	double*		ind   	=	mxGetPr(prhs[3]);
	int			nodes	=   (int) mxGetScalar(prhs[4]);
    
	double *X;    
    
	/* set up output arguments */
	plhs[0] = mxCreateDoubleMatrix(n,k,mxREAL);
    
	X = mxGetPr(plhs[0]);
	altra_mt(X, V, n, k, ind, nodes);
}

