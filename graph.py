import matplotlib.pyplot as plt
import numpy as np

# Simple donut chart
fig, ax = plt.subplots(figsize=(10, 8))

# Data
categories = ['Negative', 'Neutral', 'Positive']
f1_scores = [0.62, 0.82, 0.66]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Create donut chart
wedges, texts, autotexts = ax.pie(f1_scores, labels=categories, colors=colors,
                                 autopct='%1.2f', startangle=90,
                                 wedgeprops=dict(width=0.5))

# Style the text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

# Add title
plt.title('F1-Scores by Class\nFeedHub Classification Performance', 
          fontsize=16, fontweight='bold', pad=20)

# Add legend with all metrics
legend_text = (f'Negative: Precision=0.59, Recall=0.65, F1=0.62\n'
               f'Neutral:  Precision=0.80, Recall=0.84, F1=0.82\n'
               f'Positive: Precision=0.71, Recall=0.62, F1=0.66')

ax.text(-1.5, -1.3, legend_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", 
                                                       facecolor='lightgray', 
                                                       alpha=0.7))

plt.tight_layout()
plt.savefig('simple_f1_donut.png', dpi=300, bbox_inches='tight')
plt.show()