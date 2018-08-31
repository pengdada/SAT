#include <stdio.h>
#include <iostream>
#include "integralImage.h"


int main(int argc, char** argv)
{
	int idx = 0;
	if (argc > 1)
		idx = atoi(argv[1]);
	//using namespace BlockScan;
	extern int mainLF(int argc, char** argv);
	//mainLF(argc, argv);
#if 0
	//Test CPU Scan
	//    1, Naive-scan
	//    2, OpenMP-opmized scan
	Test();
#endif
	if (idx == 0) {
		//Test ScanRow-BRLT
		extern void TestBlockScan(int argc, char** argv);
		TestBlockScan(argc, argv);
	}
	if (idx == 1) {
		//Test BRLT-ScanRow
		extern void TestSerielScan(int argc, char** argv);
		TestSerielScan(argc, argv);
	}
	if (idx == 2) {
		//Test ScanRowColumn
		extern void TestIncreamentScan(int argc, char** argv);
		TestIncreamentScan(argc, argv);
	}
	if (idx == 3) {
		//Test ScanRowColumn Column Only
		extern void TestIncrementScanY();
		TestIncrementScanY();
	}
	if (idx == 4) {
		//Test ScanRowColumn Row only
		extern void TestIncreamentScanX();
		TestIncreamentScanX();
	}
	extern void TestSerielScanNew();
	//TestSerielScanNew();

	return 0;
}