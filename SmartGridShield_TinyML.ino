#include <OneWire.h>
#include <DallasTemperature.h>

#define PV_TEMP_PIN 2
#define AMBIENT_TEMP_PIN 3
#define LDR_PIN A0
#define PV_VOLTAGE_PIN A1
#define GREEN_LED 6
#define YELLOW_LED 7
#define RED_LED 8
#define BUZZER 9

const float TEMP_THRESHOLD=34.325;
const float VOLTAGE_THRESHOLD_1=5.2245;
const float VOLTAGE_THRESHOLD_2=4.6015;
const int LDR_THRESHOLD=323;

OneWire owPV(PV_TEMP_PIN), owAmb(AMBIENT_TEMP_PIN);
DallasTemperature pvSensor(&owPV), ambSensor(&owAmb);

float getTemp(DallasTemperature &s){
  s.requestTemperatures();
  float t=s.getTempCByIndex(0);
  if(t==DEVICE_DISCONNECTED_C || t<-40 || t>85) return NAN;
  return t;
}

float getPVVoltage(){
  int adc=analogRead(PV_VOLTAGE_PIN);
  return (adc*5.0/1023.0)*2.0; // 10k/10k divider
}

String classify(float pvT,float ambT,int ldr,float v){
  if(isnan(pvT)||isnan(ambT)||v<0||v>10) return "SENSOR_ERROR";
  if(pvT>=TEMP_THRESHOLD) return "THERMAL_ANOMALY";
  if(v>=VOLTAGE_THRESHOLD_1) return "HEALTHY";
  if(v>=VOLTAGE_THRESHOLD_2) return "SOILING";
  if(ldr<LDR_THRESHOLD) return "SOILING";
  return "PARTIAL_SHADING";
}

void alert(String s){
  digitalWrite(GREEN_LED,LOW); digitalWrite(YELLOW_LED,LOW);
  digitalWrite(RED_LED,LOW); noTone(BUZZER);
  if(s=="HEALTHY") digitalWrite(GREEN_LED,HIGH);
  else if(s=="SOILING") digitalWrite(YELLOW_LED,HIGH);
  else if(s=="PARTIAL_SHADING"){digitalWrite(YELLOW_LED,HIGH);tone(BUZZER,2000,150);}
  else {digitalWrite(RED_LED,HIGH);tone(BUZZER,2000,500);}
}

void setup(){
  Serial.begin(115200);
  pvSensor.begin(); ambSensor.begin();
  pinMode(GREEN_LED,OUTPUT); pinMode(YELLOW_LED,OUTPUT);
  pinMode(RED_LED,OUTPUT); pinMode(BUZZER,OUTPUT);
  Serial.println("SMART-GRID SHIELD");
}

void loop(){
  float pvT=getTemp(pvSensor), ambT=getTemp(ambSensor);
  int ldr=analogRead(LDR_PIN);
  float v=getPVVoltage();
  float d=(isnan(pvT)||isnan(ambT))?NAN:pvT-ambT;
  String result=classify(pvT,ambT,ldr,v);
  alert(result);

  Serial.print("PV Temperature : "); if(isnan(pvT)) Serial.println("ERROR"); else {Serial.print(pvT,2);Serial.println(" C");}
  Serial.print("Ambient Temp   : "); if(isnan(ambT)) Serial.println("ERROR"); else {Serial.print(ambT,2);Serial.println(" C");}
  Serial.print("Delta T        : "); if(isnan(d)) Serial.println("ERROR"); else {Serial.print(d,2);Serial.println(" C");}
  Serial.print("LDR            : "); Serial.println(ldr);
  Serial.print("PV Voltage     : "); Serial.print(v,2); Serial.println(" V");
  Serial.print("TinyML Result  : "); Serial.println(result);
  Serial.println();
  delay(2000);
}
