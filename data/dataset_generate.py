import pandas as pd
import numpy as np
import os

# Set a random seed so you get the exact same dataset every time you run this
np.random.seed(42)

# Define dataset size
TOTAL_STUDENTS = 1000
# Simulate a 20% dropout rate (realistic for many institutions)
DROPOUT_RATE = 0.20
NUM_DROPOUTS = int(TOTAL_STUDENTS * DROPOUT_RATE)
NUM_SAFE = TOTAL_STUDENTS - NUM_DROPOUTS

print(f"Generating data for {TOTAL_STUDENTS} students...")
print(f"- {NUM_SAFE} Not at Risk (0)")
print(f"- {NUM_DROPOUTS} At Risk (1)")

# --- 1. Generate data for students NOT AT RISK (Class 0) ---
# We use normal distributions (mean, standard_deviation, size)
# and clip the values between 0 and 100 so percentages make sense.
safe_attendance = np.clip(np.random.normal(85, 10, NUM_SAFE), 0, 100)
safe_marks = np.clip(np.random.normal(75, 12, NUM_SAFE), 0, 100)
safe_assignments = np.clip(np.random.normal(88, 10, NUM_SAFE), 0, 100)
safe_tests = np.clip(np.random.normal(78, 15, NUM_SAFE), 0, 100)
safe_participation = np.clip(np.random.normal(80, 15, NUM_SAFE), 0, 100)
safe_target = np.zeros(NUM_SAFE, dtype=int)

# --- 2. Generate data for students AT RISK (Class 1) ---
# Notice the means are much lower, but standard deviation is higher 
# to simulate erratic behavior.
risk_attendance = np.clip(np.random.normal(45, 20, NUM_DROPOUTS), 0, 100)
risk_marks = np.clip(np.random.normal(40, 18, NUM_DROPOUTS), 0, 100)
risk_assignments = np.clip(np.random.normal(35, 25, NUM_DROPOUTS), 0, 100)
risk_tests = np.clip(np.random.normal(45, 20, NUM_DROPOUTS), 0, 100)
risk_participation = np.clip(np.random.normal(30, 20, NUM_DROPOUTS), 0, 100)
risk_target = np.ones(NUM_DROPOUTS, dtype=int)

# --- 3. Combine the two groups ---
attendance = np.concatenate([safe_attendance, risk_attendance])
marks = np.concatenate([safe_marks, risk_marks])
assignments = np.concatenate([safe_assignments, risk_assignments])
tests = np.concatenate([safe_tests, risk_tests])
participation = np.concatenate([safe_participation, risk_participation])
dropout = np.concatenate([safe_target, risk_target])

# --- 4. Create a Pandas DataFrame ---
df = pd.DataFrame({
    'attendance': np.round(attendance, 1),
    'marks': np.round(marks, 1),
    'assignments': np.round(assignments, 1),
    'tests': np.round(tests, 1),
    'participation': np.round(participation, 1),
    'dropout': dropout
})

# --- 5. Shuffle the dataset ---
# We shuffle it so all the '0's aren't at the top and '1's at the bottom.
# frac=1 means return 100% of the rows, reset_index(drop=True) cleans up row numbers.
df = df.sample(frac=1).reset_index(drop=True)

# --- 6. Save to CSV ---
# Create the 'data' directory if it doesn't exist yet
os.makedirs('data', exist_ok=True)
csv_path = 'data/students.csv'

df.to_csv(csv_path, index=False)
print(f"✅ Success! Dataset saved to {csv_path}")
print("\nFirst 5 rows of your new dataset:")
print(df.head())