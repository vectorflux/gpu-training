#include <sys/time.h>
#include <stdio.h>

void beginTimer(struct timeval* time)
{
	gettimeofday(time, NULL); 
}


// returns the time in MILLISECONDS
double endTimer(struct timeval* time)
{

	struct timeval tv;
	gettimeofday(&tv, NULL);

	double elapsedTime = (tv.tv_sec)*1000000. + tv.tv_usec;
	//printf("elpased: %g ", elapsedTime);
	elapsedTime -= time->tv_sec*1000000. + time->tv_usec;      // sec to ms
	//printf("%g ", elapsedTime/1000.);

	return elapsedTime/1000.;
}
