import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import joblib
import os

class DynamicPricingModel:

    def __init__(self, model_path='dynamic_pricing_model.pkl', 
                 vehicle_encoder_path='vehicle_encoder.pkl'):
        self.model_path = model_path
        self.vehicle_encoder_path = vehicle_encoder_path
        self.model = None
        self.vehicle_encoder = None
        self.is_trained = False
    
    def _extract_features(self, df):
    
        df = df.copy()
        
        # Parse timestamps
        df['timeIn'] = pd.to_datetime(df['timeIn'], format='%d-%m-%Y %H:%M')
        df['timeOut'] = pd.to_datetime(df['timeOut'], format='%d-%m-%Y %H:%M')
        
        # Extract temporal features
        df['hour_of_day'] = df['timeIn'].dt.hour
        df['day_of_week'] = df['timeIn'].dt.dayofweek
        df['duration_minutes'] = (df['timeOut'] - df['timeIn']).dt.total_seconds() / 60
        
        # Clip duration to reasonable bounds (max 1440 minutes = 24 hours)
        df['duration_minutes'] = df['duration_minutes'].clip(lower=0, upper=1440)
        
        return df
    
    def _calculate_multiplier(self, bill_amt, base_rate=50):

        # Keep legacy behaviour if needed (not used in new training)
        multiplier = bill_amt / base_rate
        return np.clip(multiplier, 1.0, 1.6)

    def _rush_score(self, hour: float) -> float:
        """
        Compute a rush score [0,1] based on hour of day.
        Creates two peaks: morning (~9) and evening (~18).
        """
        # gaussian-like peaks around 9 and 18
        # use sigma ~1.5 to widen the peaks across the window
        sigma = 1.5
        m1 = np.exp(-0.5 * ((hour - 9.0) / sigma) ** 2)
        m2 = np.exp(-0.5 * ((hour - 18.0) / sigma) ** 2)
        # maximum possible sum is 2.0 (when hour==9 and hour==18 simultaneously impossible)
        score = (m1 + m2) / 2.0
        # clip to [0,1]
        return float(np.clip(score, 0.0, 1.0))
    
    def train(self, csv_path):
        """
        Train the Random Forest model on the provided dataset.
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Extract features
        df = self._extract_features(df)

        # Remove rows with missing exit times or vehicles that haven't exited yet
        if 'status' in df.columns:
            before = len(df)
            df = df[df['status'].str.lower() == 'exited']
            after = len(df)
            if before != after:
                print(f"Filtered dataset: removed {before-after} rows where status!='Exited'")

        # Drop rows that have NaN durations (unparsable timeOut/timeIn)
        before = len(df)
        df = df.dropna(subset=['duration_minutes'])
        after = len(df)
        if before != after:
            print(f"Filtered dataset: removed {before-after} rows with invalid durations")
   
        df['duration_hours'] = df['duration_minutes'] / 60.0
        df['duration_component'] = 0.3 * np.clip(df['duration_minutes'] / 120.0, 0.0, 1.0)
        
        # rush score in [0,1]
        df['rush_score'] = df['hour_of_day'].apply(self._rush_score)
        # rush component: up to +0.50 at peak
        df['rush_component'] = 0.5 * df['rush_score']

        # base multiplier start at 1.0; combine components
        df['multiplier'] = 1.0 + df['duration_component'] + df['rush_component']
        # clip to target range
        df['multiplier'] = df['multiplier'].clip(lower=1.0, upper=1.6)
        
        # Encode vehicle type
        self.vehicle_encoder = LabelEncoder()
        df['vehicle_type_encoded'] = self.vehicle_encoder.fit_transform(df['vehicleType'])

        # Select features and target. Include the computed rush_score so the
        # model directly observes the time-based hotspot signal.
        X = df[['hour_of_day', 'day_of_week', 'duration_minutes', 'vehicle_type_encoded', 'rush_score']]
        y = df['multiplier']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Training Complete!")
        print(f"Test Set R² Score: {r2:.4f}")
        print(f"Test Set MSE: {mse:.4f}")
        print(f"Feature Importances:")
        for feature, importance in zip(
            ['hour_of_day', 'day_of_week', 'duration_minutes', 'vehicle_type'],
            self.model.feature_importances_
        ):
            print(f"  {feature}: {importance:.4f}")
        
        self.is_trained = True
        
        # Save model and encoder
        self.save()
    
    def predict(self, vehicle_type, time_in, time_out, paid_amt):
        """
        Predict the price multiplier for a given input.
        
        Args:
            vehicle_type (str): Type of vehicle (e.g., 'twoWheeler', 'fourWheeler', 'heavyVehicle')
            time_in (str or datetime): Entry time in format 'DD-MM-YYYY HH:MM'
            time_out (str or datetime): Exit time in format 'DD-MM-YYYY HH:MM'
            paid_amt (float): Amount paid (used for reference, not directly in prediction)
        
        Returns:
            dict: Containing the multiplier and prediction details
        """
        if not self.is_trained and not self.load():
            raise RuntimeError("Model not trained. Please train the model first.")
        
        # Parse timestamps if strings
        if isinstance(time_in, str):
            time_in = pd.to_datetime(time_in, format='%d-%m-%Y %H:%M')
        if isinstance(time_out, str):
            time_out = pd.to_datetime(time_out, format='%d-%m-%Y %H:%M')
        
        # Extract features
        hour_of_day = time_in.hour
        day_of_week = time_in.dayofweek
        duration_minutes = (time_out - time_in).total_seconds() / 60
        duration_minutes = np.clip(duration_minutes, 0, 1440)
        # compute rush_score (same as used during training)
        rush_score = self._rush_score(hour_of_day)
        
        # Encode vehicle type
        try:
            vehicle_type_encoded = self.vehicle_encoder.transform([vehicle_type])[0]
        except ValueError:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")
        
        # Create feature vector (include rush_score)
        X = pd.DataFrame(
            [[hour_of_day, day_of_week, duration_minutes, vehicle_type_encoded, rush_score]],
            columns=['hour_of_day', 'day_of_week', 'duration_minutes', 'vehicle_type_encoded', 'rush_score']
        )
        
        # Predict multiplier
        multiplier = self.model.predict(X)[0]
        multiplier = np.clip(multiplier, 1.0, 1.6)

        # Convert numpy types to native Python types for JSON/print-friendly output
        multiplier = float(multiplier)
        multiplier_rounded = round(multiplier, 2)
        duration_minutes = float(round(duration_minutes, 2))
        hour_of_day = int(hour_of_day)
        day_of_week = int(day_of_week)
        original_charge = float(paid_amt)
        adjusted_charge = float(round(original_charge * multiplier, 2))

        return {
            'multiplier': multiplier_rounded,
            'vehicle_type': vehicle_type,
            'time_in': time_in.strftime('%d-%m-%Y %H:%M'),
            'time_out': time_out.strftime('%d-%m-%Y %H:%M'),
            'hour_of_entry': hour_of_day,
            'duration_minutes': duration_minutes,
            'day_of_week': day_of_week,
            'original_charge': original_charge,
            'adjusted_charge': adjusted_charge
        }
    
    def save(self):
        """Save the trained model and encoder to disk."""
        if self.model is None or self.vehicle_encoder is None:
            raise RuntimeError("Nothing to save. Model not trained.")
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vehicle_encoder, self.vehicle_encoder_path)
        print(f"Model saved to {self.model_path}")
        print(f"Encoder saved to {self.vehicle_encoder_path}")
    
    def load(self):
        """Load the trained model and encoder from disk."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vehicle_encoder_path):
                self.model = joblib.load(self.model_path)
                self.vehicle_encoder = joblib.load(self.vehicle_encoder_path)
                self.is_trained = True
                print(f"Model loaded from {self.model_path}")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False
    
    def get_vehicle_types(self):
        """Get the list of vehicle types the model knows about."""
        if self.vehicle_encoder is not None:
            return list(self.vehicle_encoder.classes_)
        return []


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DynamicPricingModel()
    
    # Train on dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'vehicles_dataset_5000.csv'
    )
    
    print("Training model...")
    model.train(dataset_path)
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example 1: Two wheeler during off-peak
    result1 = model.predict(
        vehicle_type='twoWheeler',
        time_in='18-09-2025 10:00',
        time_out='18-09-2025 12:00',
        paid_amt=50
    )
    print("\nExample 1 - Off-peak hours:")
    for key, value in result1.items():
        print(f"  {key}: {value}")
    
    # Example 2: Four wheeler during rush hour (evening)
    result2 = model.predict(
        vehicle_type='fourWheeler',
        time_in='18-09-2025 18:00',
        time_out='18-09-2025 20:00',
        paid_amt=100
    )
    print("\nExample 2 - Rush hour (evening):")
    for key, value in result2.items():
        print(f"  {key}: {value}")
    
    # Example 3: Heavy vehicle during morning rush
    result3 = model.predict(
        vehicle_type='heavyVehicle',
        time_in='18-09-2025 08:00',
        time_out='18-09-2025 09:30',
        paid_amt=75
    )
    print("\nExample 3 - Morning rush:")
    for key, value in result3.items():
        print(f"  {key}: {value}")

