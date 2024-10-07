import pandas as pd
import numpy as np

# Parameters for the data generation
num_samples = 1000
chapters = [5, 10, 15, 20]
quizzes = [1, 2, 3, 4]
interactivity = [0.1, 0.3, 0.5, 0.7, 0.9]

# Generate synthetic data
data = {
    'x_chapters': np.random.choice(chapters, num_samples),
    'y_quizzes': np.random.choice(quizzes, num_samples),
    'z_interactivity': np.random.choice(interactivity, num_samples),
    'p1_breaks': np.random.randint(0, 15, num_samples),
    'p2_response_speed': np.random.uniform(0, 1, num_samples),
    'p3_quiz_scores': np.random.randint(0, 100, num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add some logic to make the data more realistic
df['p1_breaks'] = np.where(df['x_chapters'] > 15, df['p1_breaks'] + 2, df['p1_breaks'])
df['p2_response_speed'] = np.where(df['z_interactivity'] > 0.5, df['p2_response_speed'] * 1.2, df['p2_response_speed'])
df['p3_quiz_scores'] = np.where(df['y_quizzes'] > 2, df['p3_quiz_scores'] * 1.1, df['p3_quiz_scores'])

# Ensure values are within reasonable ranges
df['p1_breaks'] = df['p1_breaks'].clip(0, 15)
df['p2_response_speed'] = df['p2_response_speed'].clip(0, 1)
df['p3_quiz_scores'] = df['p3_quiz_scores'].clip(0, 100)

# Save the DataFrame to an Excel file
df.to_excel('course_data.xlsx', index=False)

print("Excel file 'course_data.xlsx' created successfully.")
