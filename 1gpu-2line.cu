#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "enigma.cu"

void swapTwoElement(char* arr, int index1, int index2){
    char temp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = temp;
}

__device__ bool endingPatternCompare(char* text, int textLength, char* pattern, int patternLength){
    int start = textLength - patternLength;
    for(int i = 0; i < patternLength; i++){
        if(pattern[i] != text[start + i]){
            return false;
        }
    }
    return true;
}

__global__ void decrypt(char* inputText, size_t textLength, char* plugBoard, int* rotorCombination, bool* found){
    // printf("111");
    // if(threadIdx.x == threadIdx.y || threadIdx.x == threadIdx.z || threadIdx.y == threadIdx.z){
    //     return;
    // }
    
    char pattern[10] = {'H','E','I','L','H','I','T','L','E','R'};
    char startPoint[3];
    int rotorOrder[3];
    
    ///// Set start point
    startPoint[0] = stdAlphabet[blockIdx.x];
    startPoint[1] = stdAlphabet[blockIdx.y];
    startPoint[2] = stdAlphabet[blockIdx.z];

    ///// Select rotors
    rotorOrder[0] = rotorCombination[threadIdx.x * 3];
    rotorOrder[1] = rotorCombination[threadIdx.x * 3 + 1];
    rotorOrder[2] = rotorCombination[threadIdx.x * 3 + 2];
    
    char outputText[15];
    
    enigma_run(inputText, outputText, textLength, plugBoard, rotorOrder, startPoint);
    // bool found = endingPatternCompare(outputText, textLength, pattern, 10);
    if(endingPatternCompare(outputText, textLength, pattern, 10)){
        // printf("Found:StartPoint %c, %c, %c, RotorOrder %d, %d, %d\n", 
        //         startPoint[0], startPoint[1], startPoint[2], rotorOrder[0],rotorOrder[1], rotorOrder[2]);
        // *found = true;
    }
}

int main(){
    timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaError_t status;
    size_t textLength = 15;
    char inputText[15] = {'L','U','Q','N','H','X','V','L','X','E','H','O','R','L','X'};
    char* inputText_d;
    int* rotorCombination_d;


    status = cudaMalloc((void**)&inputText_d, textLength * sizeof(char));
    status = cudaMemcpy(inputText_d, inputText, textLength * sizeof(char), cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        exit(4);
    }

    dim3 gridSize(26, 26, 26);
    dim3 blockSize(60, 1, 1);

    char* plugBoard_d;
    status = cudaMalloc((void**)&plugBoard_d, 26 * sizeof(char));

    // char plugBoard_h[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    // status = cudaMemcpy(plugBoard_d, plugBoard_h, 26 * sizeof(char), cudaMemcpyHostToDevice);
    // if(status != cudaSuccess){
    //     exit(5);
    // }

    int rotorCombination[3 * 60];
    int count = 0;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            for(int k = 0; k < 5; k++){
                if(i != j && i != k && j != k){
                    rotorCombination[count * 3] = i;
                    rotorCombination[count * 3 + 1] = j;
                    rotorCombination[count * 3 + 2] = k;
                    count++;
                }
            }
        }
    }
    status = cudaMalloc((void**)&rotorCombination_d, 180 * sizeof(int));
    status = cudaMemcpy(rotorCombination_d, rotorCombination, 180 * sizeof(int), cudaMemcpyHostToDevice);
    
    // for(int i = 0; i < 60; i++){
    //     printf("%d %d %d\n", rotorCombination[i*3], rotorCombination[i*3+1], rotorCombination[i*3+2]);
    // }
    // printf("here\n");
    // decrypt<<<gridSize, blockSize>>>(inputText_d, textLength, plugBoard_d);
    /////////////////////////////////////////////////////////
    ////// Set plugboard
    /////////////////////////////////////////////////////////
    bool found[1];
    found[1] = false;
    bool* found_d;
    status = cudaMalloc((void**)&found_d, sizeof(bool));
    cudaMemcpy(found_d, found, sizeof(bool), cudaMemcpyHostToDevice);
    //////// Line1 //////////
    for(int a1 = 0; a1 < 26; a1++){
        for(int b1 = a1 + 1; b1 < 26; b1++){
            //////// Line2 //////////
            for(int a2 = 0; a2 < 26; a2++){
                if(a2 == a1 || a2 == b1) continue;
                for(int b2 = a2 + 1; b2 < 26; b2++){
                    if(b2 == a1 || b2 == b1) continue;
                    char plugBoard_h[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
                    swapTwoElement(plugBoard_h, a1, b1);  // Plug line 1
                    swapTwoElement(plugBoard_h, a2, b2);  // Plug line 2
                    status = cudaMemcpy(plugBoard_d, plugBoard_h, 26 * sizeof(char), cudaMemcpyHostToDevice);
                    decrypt<<<gridSize, blockSize>>>(inputText_d, textLength, plugBoard_d, rotorCombination_d, found_d);
                    // cudaMemcpyFromSymbol(&found, &dev_found, sizeof(bool), 0, cudaMemcpyDeviceToHost);
                    // cudaMemcpy(found, found_d, sizeof(bool), cudaMemcpyDeviceToHost);
                }
            }
            
        }
    }
    if(found[0]){
         // plugboard: %d %d, %d %d, %d %d\n", a1, b1, a2, b2, a3, b3);
        printf("Found\n");
    }
    else{
        printf("not found\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    printf("Total time: %d\n", endTime.tv_sec - startTime.tv_sec);

    return 0;
}