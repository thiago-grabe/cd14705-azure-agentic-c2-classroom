import matplotlib.pyplot as plt
import os

original_x = ['2025-09-01', '2025-09-02', '2025-09-03', '2025-09-04', '2025-09-05', '2025-09-06', '2025-09-07', '2025-09-08', '2025-09-09', '2025-09-10', '2025-09-11', '2025-09-12', '2025-09-13', '2025-09-14', '2025-09-15', '2025-09-16', '2025-09-17', '2025-09-18', '2025-09-19', '2025-09-20']
original_y = [542.0, 489.0, 563.0, 512.0, 0.0, 598.0, 621.0, 505.0, 0.0, 534.0, 511.0, 490.0, 523.0, 514.0, 2500.0, 527.0, 499.0, 5545.0, 488.0, 531.0]
cleaned_x = ['2025-09-01', '2025-09-02', '2025-09-03', '2025-09-04', '2025-09-06', '2025-09-07', '2025-09-08', '2025-09-10', '2025-09-11', '2025-09-12', '2025-09-13', '2025-09-14', '2025-09-16', '2025-09-17', '2025-09-19', '2025-09-20']
cleaned_y = [542.0, 489.0, 563.0, 512.0, 598.0, 621.0, 505.0, 534.0, 511.0, 490.0, 523.0, 514.0, 527.0, 499.0, 488.0, 531.0]

os.makedirs("artifacts", exist_ok=True)

fig, ax = plt.subplots()
ax.plot(original_x, original_y, color='blue', marker='o', label='Original Data')
ax.plot(cleaned_x, cleaned_y, color='green', marker='o', label='Clean Data')
ax.set_title('Original vs Clean Data - Website_Visits (data-Marketing-1.csv)')
ax.set_xlabel('Date')
ax.set_ylabel('Website_Visits')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("artifacts/data_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
