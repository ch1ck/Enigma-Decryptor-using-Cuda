#include "enigma.cuh"


__device__ int find_index(const char* arr, size_t arrSize, char target){
    for(int i = 0; i < arrSize; i++){
        if(arr[i] == target){
            return i;
        }
    }
    return -1;    
}

__device__ int rotorForwardMapping(int rotorID, char rotorStartPoint, int input){
    int outerRing_OffsetFromA = (rotorStartPoint - 'A' + input) % 26;
    char innerRing_mappedLetter = rotors[rotorID][outerRing_OffsetFromA];
    if(find_index(stdAlphabet, 26, innerRing_mappedLetter) < find_index(stdAlphabet, 26, rotorStartPoint))
        // distance from 'A' to mapped letter  +   distance from start point to 'A' = D(start point to mapped point)
        return find_index(stdAlphabet, 26, innerRing_mappedLetter) + (26 - find_index(stdAlphabet, 26, rotorStartPoint));
    else
        // distance from 'A' to mapped letter - distance from 'A' to start point = D(start point to mapped point)
        return find_index(stdAlphabet, 26, innerRing_mappedLetter) - find_index(stdAlphabet, 26, rotorStartPoint);
}

__device__ int rotorReverseMapping(int rotorID, char rotorStartPoint, int input){
    int innerRing_OffsetFromA = (rotorStartPoint - 'A' + input) % 26;
    char innerRing_letter = stdAlphabet[innerRing_OffsetFromA];
    int rotor_letterIndex = find_index(rotors[rotorID], 26, innerRing_letter);
    char outerRing_mappedLetter = stdAlphabet[rotor_letterIndex];
    if(find_index(stdAlphabet, 26, outerRing_mappedLetter) < find_index(stdAlphabet, 26, rotorStartPoint))
        // same as rotorForwardMapping()
        return find_index(stdAlphabet, 26, outerRing_mappedLetter) + (26 - find_index(stdAlphabet, 26, rotorStartPoint));
    else
        return find_index(stdAlphabet, 26, outerRing_mappedLetter) - find_index(stdAlphabet, 26, rotorStartPoint);
}

__device__ void rotate(int* rotorOrder, char* rotorStartPoint){
    ////////////////////////////////////////////////////////////////
    //// Implement the double stepping mechanism
    ////////////////////////////////////////////////////////////////
    if(turnover[rotorOrder[1]] == rotorStartPoint[1]){
        rotorStartPoint[1] = stdAlphabet[(find_index(stdAlphabet, 26, rotorStartPoint[1]) + 1) % 26];
        rotorStartPoint[2] = stdAlphabet[(find_index(stdAlphabet, 26, rotorStartPoint[2]) + 1) % 26];
    }

    if(turnover[rotorOrder[0]] == rotorStartPoint[0]){
        rotorStartPoint[1] = stdAlphabet[(find_index(stdAlphabet, 26, rotorStartPoint[1]) + 1) % 26];
    }

    rotorStartPoint[0] = stdAlphabet[(find_index(stdAlphabet, 26, rotorStartPoint[0]) + 1) % 26];
 
}

__device__ char encryptOneLetter(int* rotorOrder, char* rotorStartPoint, char* plugBoard, char letter){
    rotate(rotorOrder, rotorStartPoint);
    int rotor0_forward_input = find_index(plugBoard, 26, letter);
    int rotor0_forward_output = rotorForwardMapping(rotorOrder[0], rotorStartPoint[0], rotor0_forward_input);
    int rotor1_forward_output = rotorForwardMapping(rotorOrder[1], rotorStartPoint[1], rotor0_forward_output);
    int rotor2_forward_output = rotorForwardMapping(rotorOrder[2], rotorStartPoint[2], rotor1_forward_output);
    int reflector_output = find_index(stdAlphabet, 26, reflector[rotor2_forward_output]);
    int rotor2_reverse_output = rotorReverseMapping(rotorOrder[2], rotorStartPoint[2], reflector_output);
    int rotor1_reverse_output = rotorReverseMapping(rotorOrder[1], rotorStartPoint[1], rotor2_reverse_output);
    int rotor0_reverse_output = rotorReverseMapping(rotorOrder[0], rotorStartPoint[0], rotor1_reverse_output);
    char encryptedLetter = plugBoard[rotor0_reverse_output];

    return encryptedLetter;
}

// outputText & rotorStartPoint will be modified while running
__device__ void enigma_run(char* inputText, char* outputText, size_t textLength, char* plugBoard, int* rotorOrder, char* rotorStartPoint){
    // char startPoint[3];
    // for(int i = 0; i < 3; i++){
    //     startPoint[i] = rotorStartPoint[i];
    // }
    for(int i = 0; i < textLength; i++){
        outputText[i] = encryptOneLetter(rotorOrder, rotorStartPoint, plugBoard, inputText[i]);
    }
}

