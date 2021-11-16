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

void Crt_Sample(double* Xt_to_t1, double* p, double* Xt1, int V, int J)
{
	double cum_sum, probrnd;
	int token, table, total;
	int v,j,k;
	for ( j=0; j<J ; j++)
	{
		for ( v=0 ; v<V; v++)
		{
			cum_sum = p[v*J + j];
			if (Xt_to_t1[v*J+j] < 0.5)
				table = 0;
			else
			{
				for ( token = 1, table = 1; token<Xt_to_t1[v*J+j]; token++)
				{
					if ((((double) rand()) / RAND_MAX) <= cum_sum/(cum_sum + token))
						table++;
				}
				Xt1[v*J+j] = table;
			}
		}
	}
}
