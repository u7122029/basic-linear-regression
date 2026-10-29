from pathlib import Path

import numpy as np
import pandas as pd
rng = np.random.default_rng(42)   # fixed seed, so everyone gets the same data
N = 6_000                         # number of houses

# --- Inputs ---
# Floor area: right-skewed, median ~150 m²
sqm = rng.lognormal(mean=np.log(150), sigma=0.3, size=N)
sqm = np.clip(sqm, 40, 450).round(1)

# Bedrooms: driven by size, plus noise
bedrooms = 0.5 + sqm / 55 + rng.normal(0, 0.6, N)
bedrooms = np.clip(np.round(bedrooms), 1, 6).astype(int)

# Bathrooms: driven by bedrooms and size
bathrooms = 0.3 + 0.35 * bedrooms + sqm / 300 + rng.normal(0, 0.4, N)
bathrooms = np.clip(np.round(bathrooms), 1, 4).astype(int)

# Age: right-skewed (many newer homes, a long tail of old ones),
# with larger homes tending to be slightly newer
age = rng.gamma(shape=2.0, scale=15.0, size=N) - 0.05 * (sqm - 150)
age = np.clip(np.round(age), 0, 120).astype(int)

# --- Target: a true linear model ---
true_price = (
    50_000
    + 2_500 * sqm         # floor area is the main driver
    + 10_000 * bedrooms   # small effect once size is accounted for
    + 15_000 * bathrooms
    - 1_500 * age
)

# Noise that grows with size (visible in a residual plot)
noise = rng.normal(0, 30_000 + 100 * sqm)

# ~1% genuine outliers (unusual sales, data-entry errors)
is_outlier = rng.random(N) < 0.01
noise += is_outlier * rng.normal(0, 150_000, N)

price = np.clip(true_price + noise, 50_000, None).round(-3)  # nearest $1,000

df = pd.DataFrame({
    "sqm": sqm,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "price": price,
})

dataset = df.to_numpy()   # the same table as a plain (N, 5) numpy array

output_path = Path("data/house_prices.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
