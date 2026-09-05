import serial, openpyxl
PORT='COM6'; BAUD=115200; OUTPUT='SmartGridShield_Live_Data.xlsx'
ser=serial.Serial(PORT,BAUD,timeout=2)
wb=openpyxl.Workbook(); ws=wb.active; ws.title='Live Data'
ws.append(['Date','Time','PV_Temp_C','Ambient_Temp_C','DeltaT_C','LDR_ADC','PV_Voltage_V','Status'])
try:
    while True:
        line=ser.readline().decode(errors='ignore').strip()
        parts=line.split(',')
        if len(parts)==8:
            ws.append(parts); wb.save(OUTPUT); print(parts)
except KeyboardInterrupt:
    pass
finally:
    ser.close(); wb.save(OUTPUT)
