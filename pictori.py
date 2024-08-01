from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pydotplus
from collections import defaultdict

# Definim setul de date cu caracteristici despre pictori
painters_data = [
    {'Name': 'Leonardo da Vinci', 'Nationality': 'Italian', 'Period': 'Renaissance', 'Genre': 'Portrait',
     'Style': 'Realism', 'Popularity': 'Famous'},
    {'Name': 'Vincent van Gogh', 'Nationality': 'Dutch', 'Period': 'Post-Impressionism', 'Genre': 'Landscape',
     'Style': 'Expressionism', 'Popularity': 'Famous'},
    {'Name': 'Pablo Picasso', 'Nationality': 'Spanish', 'Period': 'Cubism', 'Genre': 'Abstract',
     'Style': 'Modernism', 'Popularity': 'Famous'},
    # Adăugați mai mulți pictori și caracteristicile lor aici
]

# Etichetele pentru clasificare (de exemplu, "Cunoscut", "Mai puțin cunoscut")
painters_labels = ['Famous', 'Famous', 'Famous']
# Adăugați mai multe etichete aici, una pentru fiecare pictor din painters_data

# Inițializăm un encoder de etichete pentru a transforma etichetele în valori numerice
label_encoder = LabelEncoder()

# Extragem caracteristicile și etichetele din setul de date
features = []
labels = []
for painter in painters_data:
    features.append([painter['Nationality'], painter['Period'], painter['Genre'], painter['Style']])
    labels.append(painter['Popularity'])

# Transformăm etichetele în valori numerice
encoded_labels = label_encoder.fit_transform(labels)

# Inițializăm și antrenăm modelul arborelui decizional
model = DecisionTreeClassifier(random_state=42)
model.fit(features, encoded_labels)

# Generăm arborele decizional în format DOT
dot_data = export_graphviz(model, out_file=None, feature_names=['Nationality', 'Period', 'Genre', 'Style'],
                           class_names=label_encoder.classes_, filled=True, rounded=True, special_characters=True)

# Vizualizăm arborele decizional
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")

print("Arborele decizional a fost salvat în fișierul decision_tree.png")
