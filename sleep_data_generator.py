
import pandas as pd
import numpy as np

np.random.seed(42)

data_size = 300
data = pd.DataFrame({
    'screen_time_minutes': np.random.normal(loc=180, scale=50, size=data_size).clip(0),
    'num_unlocks': np.random.poisson(lam=80, size=data_size),
    'notification_count': np.random.poisson(lam=150, size=data_size),
    'social_media_minutes': np.random.normal(loc=90, scale=30, size=data_size).clip(0),
    'work_app_minutes': np.random.normal(loc=60, scale=20, size=data_size).clip(0),
    'gaming_minutes': np.random.normal(loc=30, scale=15, size=data_size).clip(0),
    'day_of_week': np.random.randint(1, 8, size=data_size)
})

conditions = [
    (data['screen_time_minutes'] > 250) | (data['num_unlocks'] > 100),
    (data['screen_time_minutes'] < 150) & (data['notification_count'] < 120),
]
choices = ['Poor', 'Good']
data['sleep_quality'] = np.select(conditions, choices, default='Average')

data.to_csv('phone_sleep_data.csv', index=False)
