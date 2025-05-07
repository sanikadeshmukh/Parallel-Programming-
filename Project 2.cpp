#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>



using namespace std;
#define _USE_MATH_DEFINES
#define SQR(x) ((x)*(x))


// Global state variables representing the system's current condition
// starting date and time:
int NowYear = 2025;        // 2025-2030
int NowMonth = 0;          // 0 - 11 (Jan - Dec)

float NowPrecip;           // inches of rain per month (used in grain growth)
float NowTemp;             // temperature in degrees Fahrenheit for this month (used in grain growth)
float NowHeight = 5.0;     // grain height in inches
// starting state 
int NowNumDeer = 2;        // number of deer in the current population
int NowNumWolves = 2;      // number of wolves in the current ecosystem
int NowPests = 0;          // number of pests affecting grain fertility

// Constants for simulation dynamics
const float PI = 3.14159265f;
const float GRAIN_GROWS_PER_MONTH = 12.0;       // inches grain can grow each month under ideal conditions
const float ONE_DEER_EATS_PER_MONTH = 1.0;       // inches of grain eaten by one deer per month

const float AVG_PRECIP_PER_MONTH = 7.0;          // average monthly precipitation (inches)
const float AMP_PRECIP_PER_MONTH = 6.0;          // amplitude of seasonal variation in precipitation (inches)
const float RANDOM_PRECIP = 2.0;                 // random fluctuation in precipitation (inches)

const float AVG_TEMP = 60.0;                     // average monthly temperature (°F)
const float AMP_TEMP = 20.0;                     // amplitude of seasonal variation in temperature (°F)
const float RANDOM_TEMP = 10.0;                  // random fluctuation in temperature (°F)

const float MIDTEMP = 40.0;                      // ideal temperature for grain growth (°F)
const float MIDPRECIP = 10.0;                    // ideal precipitation for grain growth (inches)

unsigned int seed = 0;
omp_lock_t Lock;
volatile int NumInThreadTeam, NumAtBarrier, NumGone;

// Function to generate random float within range
float Ranf(float low, float high) {
    float r = (float)rand();
    return low + r * (high - low) / (float)RAND_MAX;
}

// Barrier synchronization functions
void InitBarrier(int n) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

void WaitBarrier() {
    // DoneComputing barrier - ensures all threads have finished computing
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if (NumAtBarrier == NumInThreadTeam) {
            NumGone = 0;
            // let all other threads get back to what they were doing
            // before this one unlocks, knowing that they might immediately
            // call WaitBarrier( ) again:
            NumAtBarrier = 0;
            while (NumGone != NumInThreadTeam - 1);
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);

    // DoneAssigning barrier - ensures all threads have updated global state
    while (NumAtBarrier != 0);  // this waits for the nth thread to arrive

    // DonePrinting barrier - ensures Watcher has printed output and advanced time
#pragma omp atomic
    NumGone++;  // this flags how many threads have returned
}

// Thread function for simulating deer population dynamics
// Deer eat grain and are affected by wolves. Carrying capacity is based on grain height.
void Deer() {
    while (NowYear < 2031) {
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)(NowHeight);
        if (nextNumDeer < carryingCapacity) nextNumDeer++;
        else if (nextNumDeer > carryingCapacity) nextNumDeer--;
        if (NowNumWolves >= nextNumDeer) nextNumDeer = 0;
        else nextNumDeer -= NowNumWolves;
        if (nextNumDeer < 0) nextNumDeer = 0;

        WaitBarrier(); // DoneComputing
        NowNumDeer = nextNumDeer;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

// Thread function for simulating grain growth dynamics
// Grain grows based on temperature and precipitation. It is eaten by deer and reduced by pests.
void Grain() {
    while (NowYear < 2031) {
        float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.));
        float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.));
        float pestFactor = 1.0 - (NowPests * 0.1);
        if (pestFactor < 0.0) pestFactor = 0.0;

        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH * pestFactor;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        if (nextHeight < 0.) nextHeight = 0.;

        WaitBarrier(); // DoneComputing
        NowHeight = nextHeight;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

// Thread function for simulating wolf population dynamics
// Wolves increase if there are deer and decrease if there are not enough deer to hunt.
void Wolves() {
    while (NowYear < 2031) {
        int nextWolves = NowNumWolves;
        if (NowNumDeer < 2) nextWolves--;
        else nextWolves++;
        if (nextWolves < 0) nextWolves = 0;
        if (nextWolves > 10) nextWolves = 10;

        WaitBarrier(); // DoneComputing
        NowNumWolves = nextWolves;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

// Thread function for simulating pest impact
// Pests grow in hot weather and when grain is abundant. They reduce crop fertility.
void Pests() {
    while (NowYear < 2031) {
        int nextPests = NowPests;
        if (NowTemp > 75.0 && NowHeight > 10.0) nextPests++;
        else if (NowTemp < 40.0 || NowHeight < 5.0) nextPests--;
        if (nextPests < 0) nextPests = 0;
        if (nextPests > 5) nextPests = 5;

        WaitBarrier(); // DoneComputing
        NowPests = nextPests;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

// Thread function responsible for managing time, weather, and printing output
// It calculates the new temperature and precipitation each month and prints all state variables.
// Thread function responsible for managing time, weather, and printing output
// It calculates the new temperature and precipitation each month and prints all state variables.
void Watcher() {
    std::ofstream outFile("simulation_output.csv");
    outFile << "Month,Year,Temp_C,Precip_cm,Deer,Grain_cm,Wolves,Pests\n";

    // Print header once to console
    printf(" Month  Year   Temp (F)   Precip (in)   Deer   Grain (in)  Wolves  Pests\n");
    printf("-------------------------------------------------------------------------\n");

    while (NowYear < 2031) {
        WaitBarrier(); // DoneComputing
        WaitBarrier(); // DoneAssigning

        // Console output
        printf("%6d %6d %10.2f %12.2f %6d %12.2f %8d %6d\n",
            NowMonth + 1, NowYear, NowTemp, NowPrecip,
            NowNumDeer, NowHeight, NowNumWolves, NowPests);

        // File output (converted units)
        float tempC = (NowTemp - 32) * 5.0 / 9.0;
        float precipCm = NowPrecip * 2.54;
        float grainCm = NowHeight * 2.54;

        outFile << (NowMonth + 1) << "," << NowYear << ","
            << std::fixed << std::setprecision(2)
            << tempC << "," << precipCm << ","
            << NowNumDeer << "," << grainCm << ","
            << NowNumWolves << "," << NowPests << "\n";

        // Advance simulation time
        NowMonth++;
        if (NowMonth == 12) {
            NowMonth = 0;
            NowYear++;
        }

        // Update weather
        float ang = (30. * (float)NowMonth + 15.) * (PI / 180.);
        float temp = AVG_TEMP - AMP_TEMP * cos(ang);
        NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
        NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
        if (NowPrecip < 0.) NowPrecip = 0.;

        WaitBarrier(); // DonePrinting
    }

    outFile.close(); // ✅ close after all data is written
}

int main() {
    omp_set_num_threads(5);
    InitBarrier(5);

#pragma omp parallel sections
    {
#pragma omp section 
        { 
            Deer(); 
        }
#pragma omp section 
        { 
            Grain(); 
        }
#pragma omp section 
        { 
            Wolves(); 
        }
#pragma omp section 
        { 
            Pests(); 
        }
#pragma omp section 
        { 
            Watcher(); 
        }
    }

    return 0;
}
