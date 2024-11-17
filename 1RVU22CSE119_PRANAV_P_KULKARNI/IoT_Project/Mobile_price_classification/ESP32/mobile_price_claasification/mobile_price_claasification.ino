#include <EloquentTinyML.h>
#include "MobilePriceClassify_model_esp32.h"
#define NUMBER_OF_INPUTS 26  
#define NUMBER_OF_OUTPUTS 4 


#define TENSOR_ARENA_SIZE 5*1024  

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

void setup() {
    Serial.begin(115200);
   
    ml.begin(MobilePriceClassify_model_esp32);
}

float fResult[NUMBER_OF_OUTPUTS] = {0};
float fRes = NULL;

void loop() {
    float input1[NUMBER_OF_INPUTS] = {1,0,0,1,1,0,0,1,0,1,0,1,1.5483596911931745,-1.2362297611990014,0.7548320081354454,-0.008935045246663081,0.620111216247627,1.425710356522435,-1.547445876673227,0.02026869615452839,0.2757116213968109,1.181997413205301,-0.5804763077229776,-0.5280612756351722,0.759508832792765,-1.4451225785068857}; 
    float input2[NUMBER_OF_INPUTS] = {0,1,1,0,0,1,0,1,0,1,0,1,-1.3795347982761568,0.8371115573737471,0.9197625270288421,-1.4036738154575337,-0.9935610936904357,-0.04624633152097916,1.510315912095323,-1.4764965583337306,-0.2587622860340611,-0.5565472705139002,1.3364532557364845,-0.2901958361598695,-1.0848685262942002,0.9159291560735261}; 
    float input3[NUMBER_OF_INPUTS] = {1,0,0,1,0,1,0,1,1,0,0,1,-0.010914350873050703,-0.7483847450642371,1.3595772440779,-1.4036738154575337,-0.5325118622795606,1.3407897783660843,-1.547445876673227,0.6854976981493102,-0.2857105502742731,-1.0201591861723538,-0.6573015871856215,0.4234004822660387,1.451150342450377,1.642406612867499}; 
    float input4[NUMBER_OF_INPUTS] = {1,0,1,0,1,0,1,0,1,0,0,1,-1.0436213219649466,-0.5044622369968549,0.3150172910863875,-0.3576197377993806,-0.5325118622795606,1.623858372220587,0.1998465740516586,-0.47865305534155794,-0.7640422405380367,0.9849623490504582,1.28832175535025,0.18553504279073596,0.5289616629068944,-1.0818838501098993}; 


    initfResult(fResult);
    fRes = ml.predict(input1, fResult);
    Serial.print("\nThe output value returned for input1 is:\n");
    Serial.println(fRes);
    displayOutput(fResult);  

    initfResult(fResult);
    fRes = ml.predict(input2, fResult);
    Serial.print("\nThe output value returned for input2 is:\n");
    Serial.println(fRes);
    displayOutput(fResult);      
    Serial.println();

    initfResult(fResult);
    fRes = ml.predict(input3, fResult);
    Serial.print("\nThe output value returned for input3 is:\n");
    Serial.println(fRes);
    displayOutput(fResult);    
  
    initfResult(fResult);
    fRes = ml.predict(input4, fResult);
    Serial.print("\nThe output value returned for input4 is:\n");
    Serial.println(fRes);
    displayOutput(fResult);  

    delay(5000); // 5 milliseconds of delay between two plots
}

void initfResult(float *fResult){
  
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++){
       fResult[i] = 0.0f;
    }
} // end of displayOutput()

void displayOutput(float *fResult){
  
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++){
        Serial.print(fResult[i]);
        Serial.print(" ");
    }
} // end of displayOutput()
