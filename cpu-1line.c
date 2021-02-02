#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include "enigma.cc"

void swapTwoElement(char* arr, int index1, int index2){
    char temp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = temp;
}

bool endingPatternCompare(char* text, int textLength, char* pattern, int patternLength){
    int start = textLength - patternLength;
    for(int i = 0; i < patternLength; i++){
        if(pattern[i] != text[start + i]){
            return false;
        }
    }
    return true;
}

int main(){
    timespec startTime, endTime;
    

    size_t textLength = 15;
    char inputText[15] = {'M','A','W','W','M','Q','S','W','Q','X','W','Y','N','Q','B'};
    char pattern[10] = {'H','E','I','L','H','I','T','L','E','R'};
    char outputText[15];
    int rotorOrder[3];
    char rotorStartPoint[3];
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

    clock_gettime(CLOCK_MONOTONIC, &startTime);
    ////// Line1 //////////
    for(int a1 = 0; a1 < 26; a1++){
        for(int b1 = a1 + 1; b1 < 26; b1++){
            char plugBoard[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
            swapTwoElement(plugBoard, a1, b1);  // Plug line 1
            ///////// Select rotor order //////////
            for(int i = 0; i < 60; i++){
                rotorOrder[0] = rotorCombination[i * 3];
                rotorOrder[1] = rotorCombination[i * 3 + 1];
                rotorOrder[2] = rotorCombination[i * 3 + 2];
                ///////// Set start point //////////
                for(int s1 = 0; s1 < 26; s1++){
                    for(int s2 = 0; s2 < 26; s2++){
                        for(int s3 = 0; s3 < 26; s3++){
                            rotorStartPoint[0] = stdAlphabet[s1];
                            rotorStartPoint[1] = stdAlphabet[s2];
                            rotorStartPoint[2] = stdAlphabet[s3];
                            enigma_run(inputText, outputText, textLength, plugBoard, rotorOrder, rotorStartPoint);
                            if(endingPatternCompare(outputText, textLength, pattern, 10)){
                                printf("Found: plugBoard , startPoint %c %c %c, rotorOrder %d %d %d\n",
                                        stdAlphabet[s1], stdAlphabet[s2], stdAlphabet[s3], rotorOrder[0], rotorOrder[1], rotorOrder[2]);
                            }
                        }
                    }
                }
            }  
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    printf("Total time: %d\n", endTime.tv_sec - startTime.tv_sec);

    return 0;
}