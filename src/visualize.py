import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load dataset
data_path = os.path.join("data", "Iris.csv")
df = pd.read_csv(data_path)

# 2. Drop the 'Id' column because it is an arbitrary row index, not a flower feature
df_features = df.drop(columns=["Id"])

# 3. Generate a pairplot
# 'hue' colors the dots according to the flower species
sns.pairplot(df_features, hue="Species", markers=["o", "s", "D"])

# 4. Save the figure locally
output_path = os.path.join("notebooks", "iris_pairplot.png")
os.makedirs("notebooks", exist_ok=True)  # ensure folder exists
plt.savefig(output_path, dpi=300)
print(f"Visualization saved successfully to: {output_path}")

# Optional: Display the interactive window
plt.show()