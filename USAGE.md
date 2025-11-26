# Usage Guide - Dynamic Pricing Model

## Overview

This model predicts a price multiplier (1.0 to 1.6) for parking charges based on rush hours and vehicle type using a Random Forest Regression algorithm.

## Model Performance

- **R² Score**: 0.9887 (98.87% accuracy)
- **Training Samples**: 5000 vehicles
- **Features Used**:
  - `hour_of_day`: Hour of entry (0-23)
  - `day_of_week`: Day of week (0-6)
  - `duration_minutes`: Parking duration in minutes
  - `vehicle_type`: Type of vehicle (twoWheeler, fourWheeler, heavyVehicle)

## Installation

```bash
pip install pandas scikit-learn numpy joblib
```

## Basic Usage

### 1. Train the Model

```python
from model import DynamicPricingModel

model = DynamicPricingModel()
model.train('vehicles_dataset_5000.csv')
```

### 2. Make Predictions

```python
result = model.predict(
    vehicle_type='twoWheeler',
    time_in='18-09-2025 18:30',      # DD-MM-YYYY HH:MM format
    time_out='18-09-2025 20:15',     # DD-MM-YYYY HH:MM format
    paid_amt=50
)

print(result)
```

### Output Example

```python
{
    'multiplier': 1.45,                    # Price multiplier (1.0 - 1.6)
    'vehicle_type': 'twoWheeler',
    'time_in': '18-09-2025 18:30',
    'time_out': '18-09-2025 20:15',
    'hour_of_entry': 18,                   # Entry hour
    'duration_minutes': 105.0,             # Total parking time
    'day_of_week': 3,                      # Wednesday (0=Monday)
    'original_charge': 50,                 # Original charge
    'adjusted_charge': 72.5                # Charge after multiplier applied
}
```

## Key Features

### Rush Hour Multipliers

- **Off-peak (10:00-17:00)**: ~1.0x (no surge)
- **Evening Rush (18:00-21:00)**: ~1.4-1.6x (high surge)
- **Morning Rush (07:00-09:00)**: ~1.2-1.5x (moderate surge)
- **Night (22:00-06:59)**: ~1.1-1.3x (slight surge)

### Vehicle Type Impact

- **Heavy Vehicle**: Higher base multipliers
- **Four Wheeler**: Medium multipliers
- **Two Wheeler**: Lower multipliers

## API Methods

### `train(csv_path)`

Trains the model on the provided CSV dataset.

### `predict(vehicle_type, time_in, time_out, paid_amt)`

Returns a dictionary with:

- `multiplier`: Final price multiplier
- `adjusted_charge`: Original charge × multiplier
- Additional metadata about the prediction

### `save()`

Saves the trained model and encoder to disk.

### `load()`

Loads a previously trained model from disk.

### `get_vehicle_types()`

Returns the list of vehicle types the model recognizes.

## Integration Example

```python
from model import DynamicPricingModel

# Load existing model
model = DynamicPricingModel()
model.load()

# Process parking entry
def calculate_dynamic_price(vehicle_type, entry_time, exit_time, base_price):
    result = model.predict(vehicle_type, entry_time, exit_time, base_price)
    return result['adjusted_charge']

# Example
final_price = calculate_dynamic_price(
    'fourWheeler',
    '18-09-2025 18:00',
    '18-09-2025 20:00',
    100
)
print(f"Final charge: ₹{final_price}")
```

## Model Details

### Feature Importance (from training):

- Vehicle Type: **69.87%** (most important)
- Duration: **29.72%**
- Hour of Day: **0.26%**
- Day of Week: **0.15%**

This shows that vehicle type is the strongest predictor, followed by parking duration.

## Notes

- Multiplier is always clipped between 1.0 and 1.6
- Duration is capped at 1440 minutes (24 hours)
- The model automatically handles timestamp parsing
- Invalid vehicle types will raise a `ValueError`
