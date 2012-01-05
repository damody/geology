#include <nmea/nmea.h>
#include <string.h>
#include <stdio.h>

int main()
{
	const char *buff[] = {
		"$GPGGA,000000,2508.644,N,12145.244,E,8,00,99.9,00000,M,00000,M,,*4A\r\n\r\n",
		"$SDDBT,0008.2,f,0002.5,M,0001.4,F*3E\r\n\r\n"
	};
	const char *buff2[] = {
		"$GPDTM,W84,,0.0000,N,0.0000,E,0.0000,W84*5F\r\n",
		"$GPAPB,V,V,-0.00,L,N,V,V,090.2,T,0009,090.2,T,090.2,T,N*6F\r\n",
		"$GPBOD,090.2,T,090.2,M,0009,0000*4E\r\n",
		"$GPBWC,000000,2508.956,N,12146.687,E,090.2,T,090.2,M,000.27,N,N*74\r\n",
		"$GPGGA,000000,2508.957,N,12146.385,E,0,00,99.9,00000,M,00000,M,,*40\r\n",
		"$GPGBS,,,,,,,,*41\r\n",
		"$GPRMC,000000,V,2508.957,N,12146.385,E,00.0,000.0,010105,,,N*57\r\n",
		"$GPVTG,000.0,T,,,00.0,N,00.0,K,N*4F\r\n",
		"$GPRMC,000000,V,2515.791,N,12145.337,E,00.0,251.7,010105,,,S*49\r\n"
	};

	int it;


	for(it = 0; it < 2; ++it)
	{
		nmeaINFO info;
		nmeaPARSER parser;

		nmea_zero_INFO(&info);
		nmea_parser_init(&parser);
		nmea_parse(&parser, buff[it], (int)strlen(buff[it]), &info);
		printf("parse: %s\n", buff[it]);
		printf("declination: %f\n", info.declination);
		printf("utc.min: %f\n", info.utc.min);
		printf("HDOP: %f\n", info.HDOP);
		printf("VDOP: %f\n", info.VDOP);
		printf("lat: %f\n", info.lat);
		printf("lon: %f\n", info.lon);
		printf("elv: %f\n", info.elv);
		printf("depth: %f\n", info.depthinfo.depth_M);
		nmea_parser_destroy(&parser);
	}



	return 0;
}
