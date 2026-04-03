#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_BMI270_BMM150.h>

Adafruit_MPU6050 mpu;

float currentThumbCurl = 0;
float currentThumbSweep = 0;
const int indexFlat = 30;   
const int middleFlat = 70;  
const int ringFlat = 70;    
const int pinkyFlat = 320;  

//boot process (checks if all of the components are connected, if not outputs which are not)
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize Nano internal IMU");
    while (1);
  }

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1);
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.println("Both IMUs Ready.");
}

void loop(){
  int rawIndex = analogRead(A7); 
  int rawMiddle = analogRead(A1);
  int rawRing = analogRead(A2);
  int rawPinky = analogRead(A3); 

  calculateThumb();
  int indexState = getGoodFingerState(rawIndex, indexFlat);
  int middleState = getGoodFingerState(rawMiddle, middleFlat);
  int ringState = getGoodFingerState(rawRing, ringFlat);
  int pinkyState = getPinkyState(rawPinky, pinkyFlat);

  char currentLetter = detectLetter(indexState, middleState, ringState, pinkyState);

  Serial.print("Letter: [ "); Serial.print(currentLetter); Serial.print(" ]");
  Serial.print(" | Index: "); Serial.print(indexState); Serial.print(" | Raw: "); Serial.print(rawIndex);
  Serial.print(" | Middle: "); Serial.print(middleState); Serial.print(" | Raw: "); Serial.print(rawMiddle);
  Serial.print(" | Ring: "); Serial.print(ringState); Serial.print(" | Raw: "); Serial.print(rawRing);
  Serial.print(" | Pinky: "); Serial.println(pinkyState); Serial.print(" | Raw: "); Serial.print(rawPinky);
  
  delay(150);
}
//checks the data for the letters
char detectLetter(int i, int m, int r, int p){
  if (i == 2 && m == 2 && r == 2 && p == 1){
    if (currentThumbCurl > -40 && currentThumbCurl < 10 && 
        currentThumbSweep < 20 && currentThumbSweep > -40){
      return 'A';
    }
  }

  if (i == 0 && m == 0 && r == 0 && p == 0){
    if (currentThumbCurl > -135 && currentThumbCurl < -55 && 
        currentThumbSweep > 20 && currentThumbSweep < 65){
      return 'B';
    }
  }
  if (i == 1 && m == 1 && r == 1 && p == 1){
    if (currentThumbCurl > -100 && currentThumbCurl < -60 && 
        currentThumbSweep < 45 && currentThumbSweep > 20){
      return 'C';
    }
  }
  if (i == 0 && m == 2 && r == 2 && p == 1){
    if (currentThumbCurl > -100 && currentThumbCurl < -60 && 
        currentThumbSweep < 45 && currentThumbSweep > 20){
      return 'D';
    }
  }
  if (i == 2 && m == 2 && r == 2 && p == 2){
    if (currentThumbCurl > -135 && currentThumbCurl < -55 && 
        currentThumbSweep > -10 && currentThumbSweep < 30){
      return 'E';
    }
  }
  if (i == 1 && m == 0 && r == 0 && p == 0){
    if (currentThumbCurl > -100 && currentThumbCurl < -60 && 
        currentThumbSweep < 45 && currentThumbSweep > 20){
      return 'F';
    }
  }
  if (i == 0 && m == 2 && r == 2 && p == 1){
    if (currentThumbCurl > -135 && currentThumbCurl < -100 && 
        currentThumbSweep < 100 && currentThumbSweep > 70){
      return 'G';
    }
  }
  if (i == 0 && m == 0 && r == 2 && p == 1){
    if (currentThumbCurl > -135 && currentThumbCurl < -100 && 
        currentThumbSweep < 100 && currentThumbSweep > 70){
      return 'H';
    }
  }
  if (i == 2 && m == 2 && r == 2 && p == 0){
    if (currentThumbCurl > -55 && currentThumbCurl < -20 && 
        currentThumbSweep < 70 && currentThumbSweep > 55){
      return 'I';
    }
  }
  if (i == 0 && m == 0 && r == 2 && p == 1){
    if (currentThumbCurl > -110 && currentThumbCurl < -90 && 
        currentThumbSweep < 75 && currentThumbSweep > 50){
      return 'K';
    }
  }
  if (i == 2 && m == 0 && r == 0 && p == 0){
    if (currentThumbCurl > -45 && currentThumbCurl < -15 && 
        currentThumbSweep < 30 && currentThumbSweep > 0){
      return 'L';
    }
  }
  if (i == 1 && m == 2 && r == 2 && p == 1){
    if (currentThumbCurl > -100 && currentThumbCurl < -60 && 
        currentThumbSweep < 25 && currentThumbSweep > 0){
      return 'T';
    }
  }

  return '-'; 
}

//calculates the angle of the thumb and uses that to know which sign it is
void calculateThumb(){
  float handX, handY, handZ;
  if (IMU.accelerationAvailable()){
    IMU.readAcceleration(handX, handY, handZ);
  }

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  float thumbX = a.acceleration.x / 9.81;
  float thumbY = a.acceleration.y / 9.81;
  float thumbZ = a.acceleration.z / 9.81;

  float handCurl = atan2(handX, handZ) * 180.0 / PI; //up down movement of the hand
  float handSweep = atan2(handY, handZ) * 180.0 / PI; //side to side movement of the hand

  float thumbCurl = atan2(thumbX, thumbZ) * 180.0 / PI; //same as the variables above, but the curl is the curl inward of the thumb
  float thumbSweep = atan2(thumbY, thumbZ) * 180.0 / PI;
  

  currentThumbCurl = thumbCurl-handCurl;
  currentThumbSweep = thumbSweep-handSweep;

  Serial.print("Thumb Curl: "); Serial.print(currentThumbCurl, 0); 
  Serial.print(" | Thumb Sweep: "); Serial.print(currentThumbSweep, 0);
}
//calculates the difference between the raw and flat value, and if above certain values returns 1 of 4 states.
int getGoodFingerState(int rawVal, int flatVal) {
  int diff = abs(rawVal-flatVal);

  if (diff<25){
    return 0;
  } else if (diff<60){
    return 1;
  } else return 2;         
}

//sensor is slightly crappy, so made the logic binary for the pinky
int getPinkyState(int rawVal, int flatVal){
  int diff = abs(rawVal - flatVal);
  int bent = 0;
  if (diff>20){
    bent = 1;
    return bent;
  } else{
    return bent;
  }
}