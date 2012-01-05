/*
 *
 * NMEA library
 * URL: http://nmea.sourceforge.net
 * Author: Tim (xtimor@gmail.com)
 * Licence: http://www.gnu.org/licenses/lgpl.html
 * $Id: parse.h 4 2007-08-27 13:11:03Z xtimor $
 *
 */

#ifndef __NMEA_PARSE_H__
#define __NMEA_PARSE_H__

#include "sentence.h"

#define NEMA_PARSE(x) \
	int nmea_parse_##x(const char *buff, int buff_sz, nmea##x *pack); \
	void nmea_##x##2info(nmea##x *pack, nmeaINFO *info);

#define NEMA_PARSE_FUNCTION(x) int nmea_parse_##x(const char *buff, int buff_sz, nmea##x *pack)
#define NEMA_PARSE_CHECK(x) NMEA_ASSERT(buff && pack); \
	memset(pack, 0, sizeof(nmea##x)); \
	nmea_trace_buff(buff, buff_sz);

#ifdef  __cplusplus
extern "C" {
#endif

int nmea_pack_type(const char *buff, int buff_sz);
int nmea_find_tail(const char *buff, int buff_sz, int *res_crc);

int nmea_parse_GPGGA(const char *buff, int buff_sz, nmeaGPGGA *pack);
int nmea_parse_GPGSA(const char *buff, int buff_sz, nmeaGPGSA *pack);
int nmea_parse_GPGSV(const char *buff, int buff_sz, nmeaGPGSV *pack);
int nmea_parse_GPRMC(const char *buff, int buff_sz, nmeaGPRMC *pack);
int nmea_parse_GPVTG(const char *buff, int buff_sz, nmeaGPVTG *pack);

NEMA_PARSE(GPAAM)
NEMA_PARSE(GPBOD)
NEMA_PARSE(GPBWW)
NEMA_PARSE(GPGLL)
NEMA_PARSE(GPMSK)
NEMA_PARSE(PGRME)

NEMA_PARSE(SDDBK)
NEMA_PARSE(SDDBS)
NEMA_PARSE(SDDBT)
NEMA_PARSE(SDDPT)
NEMA_PARSE(SDMTW)
NEMA_PARSE(WIMWV)
NEMA_PARSE(PGRMM)
NEMA_PARSE(PGRMZ)

void nmea_GPGGA2info(nmeaGPGGA *pack, nmeaINFO *info);
void nmea_GPGSA2info(nmeaGPGSA *pack, nmeaINFO *info);
void nmea_GPGSV2info(nmeaGPGSV *pack, nmeaINFO *info);
void nmea_GPRMC2info(nmeaGPRMC *pack, nmeaINFO *info);
void nmea_GPVTG2info(nmeaGPVTG *pack, nmeaINFO *info);

#ifdef  __cplusplus
}
#endif

#endif /* __NMEA_PARSE_H__ */
