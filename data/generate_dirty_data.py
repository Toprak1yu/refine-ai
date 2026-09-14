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
        r_age = random.random()
        if r_age < 0.05:
            age = -random.randint(1, 10)
        elif r_age < 0.08:
            age = random.randint(150, 300)
        elif r_age < 0.20:
            age = None
        else:
            age = random.randint(18, 70)

        r_salary = random.random()
        if r_salary < 0.03:
            salary = random.randint(5_000_000, 20_000_000)
        elif r_salary < 0.10:
            salary = None
        else:
            salary = random.randint(35_000, 180_000)

        country = random.choice(country_variations)
        city = random.choice(cities) if random.random() > 0.15 else None

        churn = 1 if random.random() < 0.08 else 0

        data.append(
            {
                "customer_id": f"CUST_{i + 1000}",
                "name": fake.name(),
                "age": age,
                "salary": salary,
                "city": city,
                "country": country,
                "churn": churn,
            }
        )

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(data)
    df.write_csv(output_path)
    print(f"✓ Generated {n_rows} dirty records -> {output_path}")


if __name__ == "__main__":
    generate_dirty_dataset()
