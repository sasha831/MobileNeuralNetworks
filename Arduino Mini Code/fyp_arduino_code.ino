#include <SparkFun_TB6612.h>

//Define Motor Controller Pins
#define AIN1 5
#define BIN1 7
#define AIN2 4
#define BIN2 8
#define PWMA 3
#define PWMB 9
#define STBY 6

// these constants are used to allow you to make your motor configuration
// line up with function names like forward.  Value can be 1 or -1
const int offsetA = 1;
const int offsetB = 1;
const int pingPin = 12; // Trigger Pin of Ultrasonic Sensor
const int echoPin = 13; // Echo Pin of Ultrasonic Sensor

//Initialise cross function variables
bool flag = true;
int motorSpeed = 0;
char input = 'N';
int driveDirection[] = {0,0,0,0,0};
int average = 0;
int tick = 0;

//Ultrasonic Sensor Variables
long prev = 0;
long prev2 = 0;
long prev3 = 0;

// Initializing motors.  The library will allow you to initialize as many
// motors as you have memory for.  If you are using functions like forward
// that take 2 motors as arguements you can either write new functions or
// call the function more than once.
Motor motor1 = Motor(AIN1, AIN2, PWMA, offsetA, STBY);
Motor motor2 = Motor(BIN1, BIN2, PWMB, offsetB, STBY);

void setup()
{
  Serial.begin(115200);
}


void loop()
{
  //Clear Variables
  int number = 0;
  int one = 0;
  int two = 0;
  int buff = 0;

  //Check for input from Jetson Nano
  if (Serial.available()) {
    //This variables counts how long the robot is stationary for
    tick = 0;
    //Shift across directions array by one
    for(int i=0;i<4;i++){
      driveDirection[i] = driveDirection[i+1]; 
    }

    //Read from serial port
    input = Serial.read();

    //Parse through serial input to get values
    if(input=='L' | input=='R') {
      //Read out buffer values
      while(!Serial.available()){}
      buff = Serial.read();
      while(!Serial.available()){}
      buff = Serial.read();

      //Get first two decimal points
      while(!Serial.available()){}
      one = Serial.read() - '0';
      while(!Serial.available()){}
      two = Serial.read() - '0';

      //Throw out extra values
      do {
        buff = Serial.read();
      } while(buff!=';');

      //Convert input into motor speed
      if (one > 0 && one < 10 && two > 0 && two < 10) {
        number = one*10 + two;
      }
    }

    // Set averages based on driving direction
    if (input == 'L') {
      average = -number;
    } else if (input == 'F') {
      average = 0;
    } else if (input == 'R'){
      average = number;
    }

    //Add averages to driving direction and calculate motor speeed
    average = 0;
    for(int i=0;i<5;i++){
      average = average + driveDirection[i];
    }
    average = average/5;
    motorSpeed = map(average,10,100,200,255);*/
  } else {
    tick++;
  }

  //Stay still if no face
  if (input == 'N') {
    motorSpeed = 0;//round(motorSpeed * 0.95);
  } else {
    motorSpeed = 200;
  }

  //Driving conditions based on speed and tick
  

  if (millis()<initTime+400) {
    //This switch statement can be used in place of the if statement
    //It provides a more robust simpler algorithm in case there is severe lag
    //it will not consider the distance to the people, only the direction
    /*switch(trueSelection) {
      case 0:
        left(motor1, motor2, 250);
        break;
      case 1:
        back(motor1, motor2, 200);
        break;
      case 2:
        right(motor1, motor2, 250);
        break;
      case 3:
        brake(motor1,motor2);
        break;
    }*/

    //Drive the robot based on direction and speed
    if (motorSpeed < -150 | tick > 200) {
      brake(motor1,motor2);
    } else if (average < -0.1) {
      right(motor1, motor2, 240);
    } else if (average > 0.1) {
      left(motor1, motor2, 240);
    } else if (average > -0.1 && average < 0.1) {
      back(motor1,motor2,200);
    } else {
      brake(motor1,motor2);
    }
  } else if(millis()<initTime+1200) {
    brake(motor1,motor2);
  } else {
    initTime = millis();
    trueSelection = selection;
  }
  
}

//This function can be called upon to place the robot in patrol mode
void patrolMode() {
  if (ultrasonic()) {
    left(motor1, motor2, 200);
  } else {
    forward(motor1, motor2, 200);
  }
}

//This function can be called upon to place the robot in sentry mode
void sentryMode() {
    left(motor1, motor2, 250);
  delay(1200);
  brake(motor1,motor2);
  delay(500);
  right(motor1, motor2, 250);
  delay(1300);
  brake(motor1,motor2);
  delay(500);
}

// Returns true if object is closer then 30cm
bool ultrasonic() {
    long duration, inches, cm;
    pinMode(pingPin, OUTPUT);
    digitalWrite(pingPin, LOW);
    delayMicroseconds(2);
    digitalWrite(pingPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(pingPin, LOW);
    pinMode(echoPin, INPUT);
    duration = pulseIn(echoPin, HIGH);
    inches = microsecondsToInches(duration);
    cm = microsecondsToCentimeters(duration);
    delay(50);
    prev3 = prev2;
    prev2 = prev;
    prev = cm;
    
    return ((cm+prev+prev2)<100)?true:false;
}

//Function for converting microseconds to inches
long microsecondsToInches(long microseconds) {
   return microseconds / 74 / 2;
}

//Function for converting microseconds to centimetres
long microsecondsToCentimeters(long microseconds) {
   return microseconds / 29 / 2;
}
