#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "enigma.cu"

//__device__ bool dev_found;
char Alphabet[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z'};

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

__global__ void decrypt(char* inputText, size_t textLength, char* plugBoard, int* rotorCombination, bool* found, int* key){
    int patternLength = 10;
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
    
    char outputText[100];
    
    int startComparison = textLength - patternLength;
    
    #pragma unroll
    for(int i = 0; i < startComparison; i++){
        rotate(rotorOrder, startPoint);
    }
    enigma_run(inputText + startComparison, outputText, patternLength, plugBoard, rotorOrder, startPoint);

    if(endingPatternCompare(outputText, patternLength, pattern, patternLength)){
        found[0] = true;
        key[0] = rotorOrder[0];
        key[1] = rotorOrder[1];
        key[2] = rotorOrder[2];
        key[3] = blockIdx.x;
        key[4] = blockIdx.y;
        key[5] = blockIdx.z;
    }
}

int main(){
    timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaError_t status;
    size_t textLength = 15;
    char inputText[15] = {'G','R','A','U','I','D','B','G','M','P','Z','R','K','R','O'};
    char* inputText_dev1;
    char* inputText_dev2;
    char* plugBoard_dev1;
    char* plugBoard_dev2;
    char plugBoard_h[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    int* rotorCombination_dev1;
    int* rotorCombination_dev2;
    bool found[1];
    found[0] = false;
    bool* found_dev1;
    bool* found_dev2;
    int key[6];
    int* key_dev1;
    int* key_dev2;
    char plugBoardKey[26];
    bool lock = false;

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

    cudaSetDevice(0);
    status = cudaMalloc((void**)&inputText_dev1, textLength * sizeof(char));
    status = cudaMalloc((void**)&plugBoard_dev1, 26 * sizeof(char));
    status = cudaMalloc((void**)&rotorCombination_dev1, 180 * sizeof(int));
    status = cudaMemcpy(inputText_dev1, inputText, textLength * sizeof(char), cudaMemcpyHostToDevice);
    status = cudaMemcpy(plugBoard_dev1, plugBoard_h, 26 * sizeof(char), cudaMemcpyHostToDevice);
    status = cudaMemcpy(rotorCombination_dev1, rotorCombination, 180 * sizeof(int), cudaMemcpyHostToDevice);
    status = cudaMalloc((void**)&found_dev1, sizeof(bool));
    cudaMemcpy(found_dev1, found, sizeof(bool), cudaMemcpyHostToDevice);
    status = cudaMalloc((void**)&key_dev1, 6 * sizeof(int));
    status = cudaMemcpy(key_dev1, key, 6 * sizeof(int), cudaMemcpyHostToDevice);
    
    if(status != cudaSuccess){
        exit(5);
    }

    cudaSetDevice(1);
    status = cudaMalloc((void**)&inputText_dev2, textLength * sizeof(char));
    status = cudaMalloc((void**)&plugBoard_dev2, 26 * sizeof(char));
    status = cudaMalloc((void**)&rotorCombination_dev2, 180 * sizeof(int));
    status = cudaMemcpy(inputText_dev2, inputText, textLength * sizeof(char), cudaMemcpyHostToDevice);
    status = cudaMemcpy(plugBoard_dev2, plugBoard_h, 26 * sizeof(char), cudaMemcpyHostToDevice);
    status = cudaMemcpy(rotorCombination_dev2, rotorCombination, 180 * sizeof(int), cudaMemcpyHostToDevice);
    status = cudaMalloc((void**)&found_dev2, sizeof(bool));
    cudaMemcpy(found_dev2, found, sizeof(bool), cudaMemcpyHostToDevice);
    status = cudaMalloc((void**)&key_dev2, 6 * sizeof(int));
    status = cudaMemcpy(key_dev2, key, 6 * sizeof(int), cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        exit(5);
    }
    

    dim3 gridSize(26, 26, 26);
    dim3 blockSize(60, 1, 1);

    // char* plugBoard_d;
    // status = cudaMalloc((void**)&plugBoard_d, 26 * sizeof(char));




    int threadRank;
    /////////////////////////////////////////////////////////
    ////// Set plugboard
    /////////////////////////////////////////////////////////

#pragma omp parallel private(threadRank) num_threads(2)
    {
    threadRank = omp_get_thread_num();
    printf("current thread: %d\n", threadRank);
    cudaSetDevice(threadRank);
    
    if(threadRank == 0){
    //////// Line1 //////////
        for(int a1 = 0; a1 < 7; a1++){
            for(int b1 = a1 + 1; b1 < 26; b1++){
                //////// Line2 //////////
                for(int a2 = 0; a2 < 26; a2++){
                    if(a2 == a1 || a2 == b1) continue;
                    for(int b2 = a2 + 1; b2 < 26; b2++){
                        if(b2 == a1 || b2 == b1) continue;
                        char plugBoard[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
                        swapTwoElement(plugBoard, a1, b1);  // Plug line 1
                        swapTwoElement(plugBoard, a2, b2);  // Plug line 2
                        status = cudaMemcpy(plugBoard_dev1, plugBoard, 26 * sizeof(char), cudaMemcpyHostToDevice);
                        decrypt<<<gridSize, blockSize>>>(inputText_dev1, textLength, plugBoard_dev1, rotorCombination_dev1, found_dev1, key_dev1);
                        cudaMemcpy(found, found_dev1, sizeof(bool), cudaMemcpyDeviceToHost);
                        status = cudaMemcpy(key, key_dev1, 6 * sizeof(int), cudaMemcpyDeviceToHost);
                        if(found[0]&&!lock){
                            lock = true;
                            for(int i = 0; i < 26; i++){
                                plugBoardKey[i] = plugBoard[i];
                            }
                        }            
                    }
                } 
            }
        }
    }
    if(threadRank == 1){
        for(int a1 = 7; a1 < 26; a1++){
            for(int b1 = a1 + 1; b1 < 26; b1++){
                //////// Line2 //////////
                for(int a2 = 0; a2 < 26; a2++){
                    if(a2 == a1 || a2 == b1) continue;
                    for(int b2 = a2 + 1; b2 < 26; b2++){
                        if(b2 == a1 || b2 == b1) continue;
                        char plugBoard[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
                        swapTwoElement(plugBoard, a1, b1);  // Plug line 1
                        swapTwoElement(plugBoard, a2, b2);  // Plug line 2
                        status = cudaMemcpy(plugBoard_dev2, plugBoard, 26 * sizeof(char), cudaMemcpyHostToDevice);
                        decrypt<<<gridSize, blockSize>>>(inputText_dev2, textLength, plugBoard_dev2, rotorCombination_dev2, found_dev2, key_dev2);
                        cudaMemcpy(found, found_dev2, sizeof(bool), cudaMemcpyDeviceToHost);
                        status = cudaMemcpy(key, key_dev2, 6 * sizeof(int), cudaMemcpyDeviceToHost);
                        if(found[0] && !lock){
                            lock = true;
                            for(int i = 0; i < 26; i++){
                                plugBoardKey[i] = plugBoard[i];
                            }   
                        }
                    }
                }
            }
        }
    }
    }

    if(found[0]){
        printf("Found rotor order: ");
        for(int i = 0; i < 3; i++){
            printf("%d", key[i]);
        }
        printf(" ");
        printf("start point: ");
        for(int i = 3; i < 6; i++){
            printf("%c", Alphabet[key[i]]);
        }
        printf("\nPlugboard: ");
        for(int i = 0; i < 26; i++){
            printf("%c ", plugBoardKey[i]);
        }
        printf("\n");
        
    }
    else{
        printf("not found\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    printf("Total time: %d\n", endTime.tv_sec - startTime.tv_sec);

    return 0;
}