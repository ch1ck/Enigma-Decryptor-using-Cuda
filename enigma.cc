// __device__ char* Enigma(char* input, size_t inputLength, char* plugBoard, int* rotorOrder, char* rotorStartPoint){

// }
#include<iostream>
using namespace std;

const char stdAlphabet[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z'};

const char reflector[26] = {'Y', 'R', 'U', 'H', 'Q', 'S', 'L', 'D', 'P', 'X', 'N', 'G', 'O', 'K', 'M', 'I', 'E', 'B', 'F', 'Z', 'C',
             'W', 'V', 'J', 'A', 'T'};

const char rotors[5][26] = {
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

const char turnover[5] = {'Q', 'E', 'V', 'J', 'Z'};

int find_index(const char* arr, size_t arrSize, char target){
    for(int i = 0; i < arrSize; i++){
        if(arr[i] == target){
            return i;
        }
    }
    return -1;    
}

int rotorForwardMapping(int rotorID, char rotorStartPoint, int input){
    int outerRing_OffsetFromA = (rotorStartPoint - 'A' + input) % 26;
    // cout << "outerRing_OffsetFromA: " << outerRing_OffsetFromA << endl;
    char innerRing_mappedLetter = rotors[rotorID][outerRing_OffsetFromA];
    // cout << "innerRing_mappedLetter: " << innerRing_mappedLetter << endl;
    if(find_index(stdAlphabet, 26, innerRing_mappedLetter) < find_index(stdAlphabet, 26, rotorStartPoint))
        // distance from 'A' to mapped letter  +   distance from start point to 'A' = D(start point to mapped point)
        return find_index(stdAlphabet, 26, innerRing_mappedLetter) + (26 - find_index(stdAlphabet, 26, rotorStartPoint));
    else
        // distance from 'A' to mapped letter - distance from 'A' to start point = D(start point to mapped point)
        return find_index(stdAlphabet, 26, innerRing_mappedLetter) - find_index(stdAlphabet, 26, rotorStartPoint);
}

int rotorReverseMapping(int rotorID, char rotorStartPoint, int input){
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

void rotate(int* rotorOrder, char* rotorStartPoint){
    // cout << "before : " << rotorStartPoint[2] << " " << rotorStartPoint[1] << " " << rotorStartPoint[0] << endl;
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

    // cout << "after : " << rotorStartPoint[2] << " " << rotorStartPoint[1] << " " << rotorStartPoint[0] << endl;
}

char encryptOneLetter(int* rotorOrder, char* rotorStartPoint, char* plugBoard, char letter){
    rotate(rotorOrder, rotorStartPoint);
    int rotor0_forward_input = find_index(plugBoard, 26, letter);
    // cout << rotor0_forward_input << endl;
    int rotor0_forward_output = rotorForwardMapping(rotorOrder[0], rotorStartPoint[0], rotor0_forward_input);
    // cout << rotor0_forward_output << endl;
    int rotor1_forward_output = rotorForwardMapping(rotorOrder[1], rotorStartPoint[1], rotor0_forward_output);
    // cout << rotor1_forward_output << endl;
    int rotor2_forward_output = rotorForwardMapping(rotorOrder[2], rotorStartPoint[2], rotor1_forward_output);
    // cout << rotor2_forward_output << endl << endl;

    int reflector_output = find_index(stdAlphabet, 26, reflector[rotor2_forward_output]);
    // cout << reflector_output << endl << endl;

    int rotor2_reverse_output = rotorReverseMapping(rotorOrder[2], rotorStartPoint[2], reflector_output);
    // cout << rotor2_reverse_output << endl;
    int rotor1_reverse_output = rotorReverseMapping(rotorOrder[1], rotorStartPoint[1], rotor2_reverse_output);
    // cout << rotor1_reverse_output << endl;
    int rotor0_reverse_output = rotorReverseMapping(rotorOrder[0], rotorStartPoint[0], rotor1_reverse_output);
    // cout << rotor0_reverse_output << endl;

    char encryptedLetter = plugBoard[rotor0_reverse_output];

    return encryptedLetter;
}

void enigma_run(char* inputText, char* outputText, size_t textLength, char* plugBoard, int* rotorOrder, char* rotorStartPoint){
    // char startPoint[3];
    // for(int i = 0; i < 3; i++){
    //     startPoint[i] = rotorStartPoint[i];
    // }
    for(int i = 0; i < textLength; i++){
        outputText[i] = encryptOneLetter(rotorOrder, rotorStartPoint, plugBoard, inputText[i]);
    }
}

// int main(){
//     int rotorOrder[3] = {2, 1, 0};
//     char rotorStartPoint[3] = {'V', 'D', 'H'};
//     char* plainText = "ZATVAWORBRQGRJSXZVNORWZBLORMEGRASLQLAFWXZYODVVTDHCIRDMNWOPNIXVKASIIIALOOSZXAMSYCQHYGPRLMSACGAWPCPAVZTMUUZCTJDVBUZAGFWMIVEZGBTLFIQDPPRZHDNKIPQHUGCXZM";
//     size_t textLength = 148;
//     char* cipherText = new char[textLength]();
//     char plugBoard[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
//                         'V', 'W', 'X', 'Y', 'Z'};
//     for(int i = 0; i < textLength; i++){
//         cipherText[i] = encryptOneLetter(rotorOrder, rotorStartPoint, plugBoard, plainText[i]);
//     }

//     cout << cipherText << endl;

//     return 0;
// }
