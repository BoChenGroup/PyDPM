#include<stdio.h>
#include<stdlib.h>
// Chaojie 2017_10_12
// No Check

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

void Crt_Multi_Sample(double* Xt_to_t1, double* Phi_t1, double* Theta_t1, double* Xt1_VK, double* Xt1_KJ, int V, int K, int J)
{
	double* probvec = (double*)malloc(K * sizeof(double));	
	double cum_sum, probrnd;
	int token, table, total;
	int v,j,k;
	for ( j=0; j<J ; j++)
	{
		for ( v=0 ; v<V; v++)
		{
			for(cum_sum = 0, k = 0; k<K ; k++)
			{
				cum_sum += Phi_t1[v*K + k] * Theta_t1[k*J + j]; // C index is different of Matlab
				probvec[k] = cum_sum;
			}

			if (Xt_to_t1[v*J+j] < 0.5)
				table = 0;
			else
			{
				for ( token = 1, table = 1; token<Xt_to_t1[v*J+j]; token++)
				{
					if ((((double) rand()) / RAND_MAX) <= cum_sum/(cum_sum + token))
						table++;
				}
			}

			for ( token = 0; token<table; token++)
			{
				double probrnd = ((double)(rand()) / RAND_MAX) * cum_sum;
				int Embedding_K = Binary_Search(probvec,probrnd,K);
				Xt1_VK[v*K + Embedding_K] += 1;
				Xt1_KJ[Embedding_K*J + j] += 1;
			}
		}
	}
	free(probvec);
}
