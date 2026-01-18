import pandas as pd
import numpy as np

# Generate sample sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')

df = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(100, 1000, 100) + np.sin(np.arange(100) * 0.3) * 200,
    'Revenue': np.random.randint(1000, 10000, 100),
    'Customers': np.random.randint(10, 100, 100),
    'Product_A': np.random.randint(50, 500, 100),
    'Product_B': np.random.randint(30, 300, 100),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Customer_Satisfaction': np.random.uniform(3.0, 5.0, 100)
})

df.to_csv('sample_sales_data.csv', index=False)
print("✓ Sample data created: sample_sales_data.csv")
print(f"✓ Shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")
