import joblib
import pandas as pd
import warnings

def data_manipulation(df):
  df.drop(columns=['policy_id'],inplace=True)
  df.dropna(inplace=True)
  # Converting all pieces of data in some columns to numrical. This way, if some data is numerical but obj or str in datatype, it will be converted to int or float as necessary
  df['policy_tenure'] = pd.to_numeric(df['policy_tenure'], errors='coerce')
  df['age_of_car'] = pd.to_numeric(df['age_of_car'], errors='coerce')
  df['age_of_policyholder'] = pd.to_numeric(df['age_of_policyholder'], errors='coerce')
  df['max_torque'] = df['max_torque'].str.extract(r'(\d+\.?\d*)').astype(float)
  df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
  df['displacement'] = pd.to_numeric(df['displacement'], errors='coerce')
  # Converting all yes/no columns into binary (0/1)
  boolean_columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera', 'is_front_fog_lights',
                   'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
                   'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert']
  for col in boolean_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
  return df

warnings.filterwarnings("ignore", category=UserWarning, message=".*Found unknown categories.*")

pipeline = joblib.load('pipeline.pkl')


columns = [
    'policy_id', 'policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster',
    'population_density', 'make', 'segment', 'model', 'fuel_type', 'max_torque',
    'max_power', 'engine_type', 'airbags', 'is_esc', 'is_adjustable_steering', 'is_tpms',
    'is_parking_sensors', 'is_parking_camera', 'rear_brakes_type', 'displacement',
    'cylinder', 'transmission_type', 'gear_box', 'steering_type', 'turning_radius',
    'length', 'width', 'height', 'gross_weight', 'is_front_fog_lights',
    'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger',
    'is_brake_assist', 'is_power_door_locks', 'is_central_locking', 'is_power_steering',
    'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
    'is_ecw', 'is_speed_alert', 'ncap_rating'
]

def get_user_input():
    print("Enter the following values:\n")
    data = {}
    for col in columns:
        val = input(f"{col}: ")
        data[col] = val
    return pd.DataFrame([data])

claim = pipeline.predict(get_user_input())

if claim[0] == 1:
    print('User is expected to make a claim')
else:
    print('User is not expected to make a claim')