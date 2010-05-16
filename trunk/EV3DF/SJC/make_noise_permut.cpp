#include <iostream>
#include <vector>
#include <SJC/SJCRandom.h>
#include <time.h>

#define SIZE 256

int
main(int argc, char *argv[])
{
    std::vector<int>	vals;

    for ( uint i = 0 ; i < SIZE ; i++ )
	vals.push_back(i);

    SJCRandom	r(time(0));
    r.Permute(vals);

    std::cout << "const uint SJCNoise::PERM_SIZE = " << SIZE << ";\n";
    std::cout << "const uint SJCNoise::PERM_ARRAY[" << SIZE * 2 << "] = {\n";
    for ( uint i = 0 ; i < SIZE ; i++ )
	std::cout << '\t' << vals[i] << ",\n";
    for ( uint i = 0 ; i < SIZE ; i++ )
    {
	std::cout << '\t' << vals[i];
	if ( i != SIZE - 1 )
	    std::cout << ",\n";
	else
	    std::cout << "\n";
    std::cout << "};\n";
}


