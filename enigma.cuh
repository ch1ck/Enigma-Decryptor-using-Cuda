#pragma once

#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////
//// Constant variable
//////////////////////////////////////////////////////////////
__constant__ char stdAlphabet[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z'};

__constant__ char reflector[26] = {'Y', 'R', 'U', 'H', 'Q', 'S', 'L', 'D', 'P', 'X', 'N', 'G', 'O', 'K', 'M', 'I', 'E', 'B', 'F', 'Z', 'C',
             'W', 'V', 'J', 'A', 'T'};

__constant__ char rotors[5][26] = {
    {'E', 'K', 'M', 'F', 'L', 'G', 'D', 'Q', 'V', 'Z', 'N', 'T', 'O', 'W', 'Y', 'H', 'X', 'U', 'S', 'P', 'A', 'I', 'B',
     'R', 'C', 'J'},
    {'A', 'J', 'D', 'K', 'S', 'I', 'R', 'U', 'X', 'B', 'L', 'H', 'W', 'T', 'M', 'C', 'Q', 'G', 'Z', 'N', 'P', 'Y', 'F',
     'V', 'O', 'E'},
    {'B', 'D', 'F', 'H', 'J', 'L', 'C', 'P', 'R', 'T', 'X', 'V', 'Z', 'N', 'Y', 'E', 'I', 'W', 'G', 'A', 'K', 'M', 'U',
     'S', 'Q', 'O'},
    {'E', 'S', 'O', 'V', 'P', 'Z', 'J', 'A', 'Y', 'Q', 'U', 'I', 'R', 'H', 'X', 'L', 'N', 'F', 'T', 'G', 'K', 'D', 'C',
     'M', 'W', 'B'},
    {'V', 'Z', 'B', 'R', 'G', 'I', 'T', 'Y', 'U', 'P', 'S', 'D', 'N', 'H', 'L', 'X', 'A', 'W', 'M', 'J', 'Q', 'O', 'F',
     'E', 'C', 'K'}};

__constant__ char turnover[5] = {'Q', 'E', 'V', 'J', 'Z'};


//////////////////////////////////////////////////////////////
//// Device funtion
//////////////////////////////////////////////////////////////
__device__ int find_index(const char* arr, size_t arrSize, char target);
__device__ int rotorForwardMapping(int rotorID, char rotorStartPoint, int input);
__device__ int rotorReverseMapping(int rotorID, char rotorStartPoint, int input);
__device__ void rotate(int* rotorOrder, char* rotorStartPoint);
__device__ char encryptOneLetter(int* rotorOrder, char* rotorStartPoint, char* plugBoard, char letter);
__device__ void enigma_run(char* inputText, char* outputText, size_t textLength, char* plugBoard, int* rotorOrder, char* rotorStartPoint);