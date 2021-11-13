#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"
#include "epph.h" /* This is the head file that contains the implementation of the used functions*/


/*
 Projection for sgLasso

  min  1/2 \|X - V\|_F^2 + \lambda_1 \|X\|_1 + \lambda_2 \|X\|_{p,1}

 Written by Jun Liu, January 15, 2010
 For any problem, please contact: j.liu@asu.edu
 
 */

void epsgLasso(double *X, double * normx, double * V, int k, int n, double lambda1, double lambda2, int pflag){
    int i, j, *iter_step, nn=n*k, m;
    double *v, *x;
    double normValue,c0=0, c;
    
    v=(double *)malloc(sizeof(double)*n);
    x=(double *)malloc(sizeof(double)*n);
    iter_step=(int *)malloc(sizeof(int)*2);
    
	/*
	initialize normx
	*/
	normx[0]=normx[1]=0;


    /*
     X and V are k x n matrices in matlab, stored in column priority manner
     x corresponds a row of X
	 
	 pflag=2:   p=2
	 pflag=0:   p=inf
     */
    
	/*
	soft thresholding 
	by lambda1

    the results are stored in X
	*/
	for (i=0;i<nn;i++){
		if (V[i]< -lambda1)
			X[i]=V[i] + lambda1;
		else
			if (V[i]> lambda1)
				X[i]=V[i] - lambda1;
			else
				X[i]=0;
	}
	
	/*
	Shrinkage or Truncating
	by lambda2
	*/
	if (pflag==2){
		for(i=0; i<k; i++){

			/*
			process the i-th row, and store it in v
			*/
			normValue=0;

			m=n%5;
			for(j=0;j<m;j++){
				v[j]=X[i + j*k];
			}
			for(j=m;j<n;j+=5){
				v[j  ]=X[i + j*k];
				v[j+1]=X[i + (j+1)*k ];
				v[j+2]=X[i + (j+2)*k];
				v[j+3]=X[i + (j+3)*k];
				v[j+4]=X[i + (j+4)*k];
			}
						
			m=n%5;
			for(j=0;j<m;j++){
				normValue+=v[j]*v[j];
			}
			for(j=m;j<n;j+=5){
				normValue+=v[j]*v[j]+
					       v[j+1]*v[j+1]+
						   v[j+2]*v[j+2]+
						   v[j+3]*v[j+3]+
						   v[j+4]*v[j+4];
			}

			/*
			for(j=0; j<n; j++){
				v[j]=X[i + j*k];

				normValue+=v[j]*v[j];
			}
			*/

			normValue=sqrt(normValue);

			if (normValue<= lambda2){
				for(j=0; j<n; j++)
					X[i + j*k]=0;

				/*normx needs not to be updated*/
			}
			else{

				normx[1]+=normValue-lambda2;
				/*update normx[1]*/

				normValue=(normValue-lambda2)/normValue;

				m=n%5;
				for(j=0;j<m;j++){
					X[i + j*k]*=normValue;
					normx[0]+=fabs(X[i + j*k]);
				}
				for(j=m; j<n;j+=5){
					X[i + j*k]*=normValue;
					X[i + (j+1)*k]*=normValue;
					X[i + (j+2)*k]*=normValue;
					X[i + (j+3)*k]*=normValue;
					X[i + (j+4)*k]*=normValue;

					normx[0]+=fabs(X[i + j*k])+
						      fabs(X[i + (j+1)*k])+
							  fabs(X[i + (j+2)*k])+
							  fabs(X[i + (j+3)*k])+
							  fabs(X[i + (j+4)*k]);
				}

				/*
				for(j=0; j<n; j++)
					X[i + j*k]*=normValue;
				*/
			}
		}
	}
	else{
		for(i=0; i<k; i++){
			
		    /*
			process the i-th row, and store it in v
			*/			
			normValue=0;
			for(j=0; j<n; j++){
				v[j]=X[i + j*k];

				normValue+=fabs(v[j]);
			}

			if (normValue<= lambda2){
				for(j=0; j<n; j++)
					X[i + j*k]=0;
			}
			else{
				eplb(x, &c, iter_step, v, n, lambda2, c0);

				for(j=0; j<n; j++){
					if (X[i + j*k] > c)
						X[i + j*k]=c;
					else
						if (X[i + j*k]<-c)
							X[i + j*k]=-c;
				}
			}
		}
	}

    
    free(v);
    free(x);
    free(iter_step);    
}

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* V=            mxGetPr(prhs[0]);
    int     k=     (int)  mxGetScalar(prhs[1]);
    int     n=     (int)  mxGetScalar(prhs[2]);
    double  lambda1=      mxGetScalar(prhs[3]);
    double  lambda2=      mxGetScalar(prhs[4]);
    int     pflag= (int)  mxGetScalar(prhs[5]);


    double *X;
	double *normx;
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix( k, n, mxREAL);	
    plhs[1] = mxCreateDoubleMatrix( 1, 2, mxREAL);
    
    X=mxGetPr(plhs[0]);
	normx=mxGetPr(plhs[1]);
    

	epsgLasso(X, normx, V, k, n, lambda1, lambda2, pflag);
}

