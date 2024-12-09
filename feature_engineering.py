def feature_engineering(ais_data, is_test=False):
    # Handle Missing Values
    print("Handling missing values...")
    ais_data['portId'].fillna(method='ffill', inplace=True)
    print("Missing values handled.")

    # Encoding categorical 'portId'
    print("Encoding 'portId' using Label Encoding...")
    label_encoder = LabelEncoder()
    ais_data['portId'] = label_encoder.fit_transform(ais_data['portId'])

    # Creating temporal features
    ais_data['time'] = pd.to_datetime(ais_data['time'])
    ais_data['hour_of_day'] = ais_data['time'].dt.hour
    ais_data['day_of_week'] = ais_data['time'].dt.dayofweek

    # Creating lag features for vessel movement
    print("Creating lag features for vessel movement...")
    lag_features = 5  # Increase lag features to 5
    for lag in range(1, lag_features + 1):
        ais_data[f'latitude_lag{lag}'] = ais_data.groupby('vesselId')['latitude'].shift(lag)
        ais_data[f'longitude_lag{lag}'] = ais_data.groupby('vesselId')['longitude'].shift(lag)
    
    # Creating additional features
    print("Creating additional features for vessel movement...")
    ais_data['distance_traveled'] = np.sqrt((ais_data['latitude'] - ais_data['latitude_lag1'])**2 + 
                                            (ais_data['longitude'] - ais_data['longitude_lag1'])**2)

    # Detecting jumps in vessel movement
    distance_mean = ais_data['distance_traveled'].mean()
    distance_std = ais_data['distance_traveled'].std()
    jump_threshold = distance_mean + 2 * distance_std
    ais_data['jump_detected'] = ais_data['distance_traveled'].apply(lambda x: 1 if x > jump_threshold else 0)

    # Marking Outliers
    print("Marking outliers in the dataset...")
    ais_data['outlier'] = ais_data['distance_traveled'].apply(lambda x: 1 if x > jump_threshold else 0)

    # Adding Rolling Average Features
    print("Adding rolling average features...")
    rolling_window = 3  # Rolling window of size 3
    ais_data['latitude_rolling_mean'] = ais_data.groupby('vesselId')['latitude'].transform(lambda x: x.rolling(rolling_window).mean())
    ais_data['longitude_rolling_mean'] = ais_data.groupby('vesselId')['longitude'].transform(lambda x: x.rolling(rolling_window).mean())
    ais_data.fillna(method='bfill', inplace=True)  # Fill NaNs generated due to rolling

    return ais_data
