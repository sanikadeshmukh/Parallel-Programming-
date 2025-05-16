#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <omp.h>
#include <string>

// setting the number of threads:
#ifndef NUMT
#define NUMT 1
#endif

// setting the number of capitals we want to try:
#ifndef NUMCAPITALS
#define NUMCAPITALS 5
#endif

// maximum iterations to allow looking for convergence:
#define MAXITERATIONS 100

// how many tries to discover the maximum performance:
#define NUMTRIES 30

#define CSV

struct city
{
	std::string name;
	float longitude;
	float latitude;
	int capitalnumber;
	float mindistance;
};

#include "UsCities.data"

// setting the number of cities we want to try:
#define NUMCITIES ( sizeof(Cities) / sizeof(struct city) )

struct capital
{
	std::string name;
	float longitude;
	float latitude;
	float longsum;
	float latsum;
	int numsum;
};

struct capital Capitals[NUMCAPITALS];

float Distance(int city, int capital)
{
	float dx = Cities[city].longitude - Capitals[capital].longitude;
	float dy = Cities[city].latitude - Capitals[capital].latitude;
	return sqrtf(dx * dx + dy * dy);
}

int main(int argc, char* argv[])
{
#ifdef _OPENMP
	fprintf(stderr, "OpenMP is supported -- version = %d\n", _OPENMP);
#else

	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	omp_set_num_threads(NUMT); // set the number of threads to use

	// seed the capitals (uniform selection)
	for (int k = 0; k < NUMCAPITALS; k++)
	{
		int cityIndex = k * (NUMCITIES - 1) / (NUMCAPITALS - 1);
		Capitals[k].longitude = Cities[cityIndex].longitude;
		Capitals[k].latitude = Cities[cityIndex].latitude;
	}

	double time0, time1;
	time0 = omp_get_wtime();
	for (int n = 0; n < MAXITERATIONS; n++)
	{
		// reset summations
		for (int k = 0; k < NUMCAPITALS; k++)
		{
			Capitals[k].longsum = 0.;
			Capitals[k].latsum = 0.;
			Capitals[k].numsum = 0;
		}

		

#pragma omp parallel for default(none) shared(Cities, Capitals) schedule(static)

		for (int i = 0; i < NUMCITIES; i++)
		{
			int capitalnumber = -1;
			float mindistance = 1.e+37f;

			for (int k = 0; k < NUMCAPITALS; k++)
			{
				float dist = Distance(i, k);
				if (dist < mindistance)
				{
					mindistance = dist;
					capitalnumber = k;
				}
			}

			Cities[i].capitalnumber = capitalnumber;
			Cities[i].mindistance = mindistance;

#pragma omp atomic
			Capitals[capitalnumber].longsum += Cities[i].longitude;

#pragma omp atomic
			Capitals[capitalnumber].latsum += Cities[i].latitude;

#pragma omp atomic
			Capitals[capitalnumber].numsum++;
		}

		

		// parallel update of new averages
#pragma omp parallel for default(none) shared(Capitals) schedule(static)
		for (int k = 0; k < NUMCAPITALS; k++)
		{
			if (Capitals[k].numsum > 0)
			{
				Capitals[k].longitude = Capitals[k].longsum / (float)Capitals[k].numsum;
				Capitals[k].latitude = Capitals[k].latsum / (float)Capitals[k].numsum;
			}
		}
	}
	time1 = omp_get_wtime();	
	// Performance metric
	double megaCityCapitalsPerSecond = (double)NUMCITIES * (double)NUMCAPITALS * MAXITERATIONS / (time1 - time0) / 1e6;

	// Extra credit: assign name of city closest to each capital
	for (int k = 0; k < NUMCAPITALS; k++)
	{
		float minDist = 1.e+37f;
		int closestCity = -1;
		for (int i = 0; i < NUMCITIES; i++)
		{
			float dx = Capitals[k].longitude - Cities[i].longitude;
			float dy = Capitals[k].latitude - Cities[i].latitude;
			float dist = sqrtf(dx * dx + dy * dy);
			if (dist < minDist)
			{
				minDist = dist;
				closestCity = i;
			}
		}
		Capitals[k].name = Cities[closestCity].name;
	}

	// Final capital city coordinates (and names if NUMT == 1)
	if (NUMT == 1)
	{
		for (int k = 0; k < NUMCAPITALS; k++)
		{
			fprintf(stderr, "\t%3d:  %8.2f , %8.2f , %s\n",
				k, Capitals[k].longitude, Capitals[k].latitude, Capitals[k].name.c_str());
		}
	}

#ifdef CSV
	fprintf(stdout, "%d,%d,%.3lf\n", NUMT, NUMCAPITALS, megaCityCapitalsPerSecond);
#else
	fprintf(stderr, "%2d threads : %4d cities ; %4d capitals; megatrials/sec = %8.3lf\n",
		NUMT, NUMCITIES, NUMCAPITALS, megaCityCapitalsPerSecond);
#endif
}
