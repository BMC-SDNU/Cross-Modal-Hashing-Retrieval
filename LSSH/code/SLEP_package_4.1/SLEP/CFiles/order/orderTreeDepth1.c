#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

#include "orderTree.h"



/*
 * In this file, we propose an O(n^2) algorithm for solving the problem:
 *
 * min   1/2 \|x - u\|^2
 * s.t.  x_i \ge x_j \ge 0, (i,j) \in E,
 *
 * where E is the edge set of the tree
 *
 * The tree is a tree with depth 1
 *
*/

/* 
 * We write the wrapper for calling from Matlab
 *
 * orderTreeDepth1(double *x, double *u, int n){
 *
*/

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* u=                mxGetPr(prhs[0]);
    int     n=       (int )   mxGetScalar(prhs[1]);	
    
    double *x;    
            
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix( n, 1, mxREAL); 	
    x=  mxGetPr(plhs[0]);
   
	orderTreeDepth1(x, u, n);
}

