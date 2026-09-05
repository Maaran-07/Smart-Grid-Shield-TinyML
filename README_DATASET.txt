# Smart-Grid Shield — Dataset README

## 1. Dataset Overview

This dataset is developed for the **Smart-Grid Shield** project, a low-cost
edge-based photovoltaic (PV) anomaly detection and monitoring system.

The dataset contains **600 samples** representing five PV operating conditions.
The synthetic dataset was derived using patterns and characteristics obtained
from **50 field-level observations**.

> **Important:** The 600 samples are synthetic/derived data and must not be
> presented as 600 independent field measurements.

---

## 2. Project Objective

The dataset is used to:

- Monitor important PV operating parameters.
- Identify abnormal PV operating conditions.
- Compare multiple machine-learning algorithms.
- Evaluate model performance using standard classification metrics.
- Select a lightweight model suitable for TinyML deployment on Arduino.

---

## 3. Dataset Composition

Total samples: **600**

| Condition | Samples |
|---|---:|
| Healthy | 120 |
| Partial Shading | 120 |
| Soiling | 120 |
| Thermal Anomaly | 120 |
| Sensor Anomaly | 120 |
| **Total** | **600** |

The dataset is balanced, with 120 samples for each condition.

---

## 4. Input Features

The machine-learning models use the following five features:

| Feature | Unit | Description |
|---|---|---|
| `PV_Temp_C` | °C | Temperature measured at the PV panel |
| `Ambient_Temp_C` | °C | Surrounding ambient temperature |
| `DeltaT_C` | °C | Difference between PV and ambient temperature |
| `LDR_ADC` | ADC value | Relative light-intensity measurement |
| `PV_Voltage_V` | V | PV panel voltage |

### Derived Parameters

`DeltaT_C` is calculated as:

PV Temperature − Ambient Temperature

Other recorded parameters such as estimated current and estimated power
may be included for analysis but are not necessarily used as ML inputs.

---

## 5. Dataset Columns

The primary monitoring sheet contains:

```text
Date
Time
PV_Temp_C
Ambient_Temp_C
DeltaT_C
LDR_ADC
PV_Voltage_V
Estimated_Current_mA
Estimated_Power_W
Status
Condition500 generated synthetic samples. 100 per condition. Not field data.
