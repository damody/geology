

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nmea/nmea.h>


int main()
{
	nmeaGENERATOR *gen;
	nmeaINFO info;
	char buff[2048];
	int gen_sz;
	int it;
	FILE* file;

	nmea_zero_INFO(&info);

	if(0 == (gen = nmea_create_generator(NMEA_GEN_ROTATE, &info)))
		return -1;
	
	file = fopen("go.txt", "w");
	for(it = 0; it < 100; ++it)
	{
		memset(buff, 0, 2048);
		gen_sz = nmea_generate_from(
			&buff[0], 2048, &info, gen,
			GPGGA | SDDBT //  | GPGSA | GPGSV | GPRMC | GPVTG
			);

		buff[gen_sz] = 0;
		//printf("%s\n", &buff[0]);
		fprintf(file, "%s", buff);

#ifdef NMEA_WIN
		//Sleep(500);
#else
		usleep(500000);        
#endif
	}
	fclose(file);
	nmea_gen_destroy(gen);

	return 0;
}
