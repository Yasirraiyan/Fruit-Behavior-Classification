import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
Fruit = ['Orange', 'Banana', 'Other'] 
Behavior = ['Yellow', 'Sweet', 'Long'] 
Behavior_array = np.array([[350, 450, 0], [400, 300, 350], [50, 100, 150]])
plt.xlabel('Fruit') 
plt.ylabel('Behavior Value') 
gnb = GaussianNB() 
gnb.fit(Behavior_array, [0, 1, 2]) 
print(f"Predictions:{gnb}")
plt.show()
predicted = gnb.predict(Behavior_array)
plt.xlabel('Fruit') 
plt.ylabel('Behavior Value') 
plt.scatter(Fruit,Behavior)
plt.legend() 
plt.title('Original Scatter Plot')
plt.xlabel('Fruit') 
plt.ylabel('Behavior Value') 

plt.legend() 
plt.title('Marked Area with Predictions') 
plt.tight_layout() 
plt.show() 
# Output the predictions 
print(f"Predictions: {predicted}") 
Fruit_encoded = ['Orange', 'Banana', 'Other']
decoded_predictions = [Fruit_encoded[pred]
for pred in predicted] 
# Original Scatter Plot 
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1) 
plt.scatter([Fruit[0]] * len(Behavior_array[0]), Behavior_array[0], label=Fruit[0]) 
plt.scatter([Fruit[1]] * len(Behavior_array[1]), Behavior_array[1], label=Fruit[1]) 
plt.scatter([Fruit[2]] * len(Behavior_array[2]), Behavior_array[2], label=Fruit[2]) 
plt.xlabel('Fruit') 
plt.ylabel('Behavior Value') 
plt.legend() 
plt.title('Original Scatter Plot') 
# Marked Area in Another Graph
plt.subplot(1, 2, 2)
plt.scatter([Fruit[0]] * len(Behavior_array[0]), Behavior_array[0], label=Fruit[0], alpha=0.6) 
plt.scatter([Fruit[1]] * len(Behavior_array[1]), Behavior_array[1], label=Fruit[1], alpha=0.6) 
plt.scatter([Fruit[2]] * len(Behavior_array[2]), Behavior_array[2], label=Fruit[2], alpha=0.6) 
# Highlighting the predicted areas with circles and colors 
colors = ['red', 'blue', 'green'] 
for i, fruit in enumerate(Fruit_encoded): plt.scatter([Fruit[i]] * len(Behavior_array[i]), Behavior_array[i], s=100, facecolors='none', edgecolors=colors[i], linewidths=2, label=f'Predicted {fruit}')
le = LabelEncoder()
Behavior_encoded = le.fit_transform(Behavior)
