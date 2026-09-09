import random
from pathlib import Path
import polars as pl
from faker import Faker

fake = Faker("en_US")
random.seed(42)

def generate_dirty_dataset(n_rows: int = 500, output_path: str = "data/raw/dirty_customers.csv") -> None:
    """Generates a synthetic dirty dataset with missing values, extreme outliers, and categorical noise."""
    data = []
    
    cities = ["New York", "San Francisco", "Austin", "Seattle", "Chicago"]
    country_variations = ["US", "USA", "United States", "united states", "US_OFFICIAL", None]
    
    for i in range(n_rows):
        # 1. Age Anomalies: Negative, impossible bounds, or missing
        r_age = random.random()
        if r_age < 0.05:
            age = -random.randint(1, 10)  # Negative age anomaly
        elif r_age < 0.08:
            age = random.randint(150, 300)  # Impossible age bound
        elif r_age < 0.20:
            age = None  # Missing value (triggers the 20% rule across the column)
        else:
            age = random.randint(18, 70)
            
        # 2. Salary Anomalies: Extreme outliers (|Z-score| > 3.0) or missing
        r_salary = random.random()
        if r_salary < 0.03:
            salary = random.randint(5_000_000, 20_000_000)  # Extreme outlier
        elif r_salary < 0.10:
            salary = None
        else:
            salary = random.randint(35_000, 180_000)
            
        # 3. Categorical Inconsistencies
        country = random.choice(country_variations)
        city = random.choice(cities) if random.random() > 0.15 else None
        
        # 4. Target Label (Churn): Severe class imbalance (~8% churn rate)
        churn = 1 if random.random() < 0.08 else 0
        
        data.append({
            "customer_id": f"CUST_{i+1000}",
            "name": fake.name(),
            "age": age,
            "salary": salary,
            "city": city,
            "country": country,
            "churn": churn
        })
        
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(data)
    df.write_csv(output_path)
    print(f"✓ Generated {n_rows} dirty records -> {output_path}")

if __name__ == "__main__":
    generate_dirty_dataset()