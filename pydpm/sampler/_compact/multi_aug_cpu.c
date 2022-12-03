#include<stdio.h>
#include<stdlib.h>
// Chaojie 2017_10_12

int Binary_Search(double *probvec, double prob, int K)
{
	int kstart, kend, kmid;
	// K : the length of probvec
	if (prob <= probvec[0])
		return(0);
	else
	{
		for(kstart = 1, kend = K-1;;)
		{
			if (kstart >= kend)
				return(kend);
			else
			{
				kmid = (kstart + kend)/2;
				if (probvec[kmid-1]>=prob)
					kend = kmid - 1;
				else if (probvec[kmid]<prob)
					kstart = kmid + 1;
				else
					return(kmid);
			}
		}
	}
	return(kmid);
}
	
void Multi_Sample(double* X, double* Phi, double* Theta, double* XVK, double* XKJ, int V, int K, int J)
{
	double* probvec = (double*)malloc(K * sizeof(double));
	
//	if (probvec == NULL)
//		printf("Malloc Error, No space!\n");

	for (int v=0;v<V;v++)
	{
		for (int j=0;j<J;j++)
		{
			if(X[v*J+j]<0.5)
				continue;
			else
			{
				double cumsum = 0.0;
				for(int k=0;k<K;k++)
				{
					cumsum += Phi[v*K + k] * Theta[k*J + j];
					probvec[k] = cumsum;
				}

				for (int token = 0; token<X[v*J + j]; token++)
				{
					double probrnd = ((double)(rand()) / RAND_MAX) * cumsum;
					int Embedding_K = Binary_Search(probvec,probrnd,K);
					XVK[v*K + Embedding_K] += 1;
					XKJ[Embedding_K*J + j] += 1;
				}
			}		
		}
	}
	free(probvec);
}

void Multi_Input(double* X, double* Pro1, double* Pro2, double* X_t_1, double* X_t_2, int V, int J)
{
	double* probvec = (double*)malloc(2 * sizeof(double));
	for (int v=0;v<V;v++)
	{
		for (int j=0;j<J;j++)
		{
			if(X[v*J+j]<0.5)
				continue;
			else
			{
				double cumsum = 0.0;
				cumsum += Pro1[v*J+j];
				probvec[0] = cumsum;
				cumsum += Pro2[v*J+j];
				probvec[1] = cumsum;

				for (int token = 0; token<X[v*J + j]; token++)
				{
					double probrnd = ((double)(rand()) / RAND_MAX) * cumsum;
					int Embedding_K = Binary_Search(probvec,probrnd,2);
					if(Embedding_K == 0)
						X_t_1[v*J + j]+=1;
					else
						X_t_2[v*J + j]+=1;
				}
			}		
		}
	}
	free(probvec);
}

