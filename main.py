import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("heart.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.shape)

x=df.drop("target",axis=1)
y=df['target']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#train the decision tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
train_acc = dt.score(X_train,y_train)
test_acc = dt.score(X_test,y_test)
print("training Accuracy:",train_acc)
print("Testing Accuracy:",test_acc)

#visualize the tree
plt.figure(figsize=(20,10))

tree.plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["No Disease", "Disease"],
    filled=True
)
plt.savefig("visualize_tree.png")
plt.show()

#analyze overfiting
print("\nDepth Analysis")

for depth in [2, 3, 4, 5, 6]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(
        f"Depth={depth} | "
        f"Train={train_acc:.3f} | "
        f"Test={test_acc:.3f}"
    )

# train random forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("\nRandom Forest")
print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))  

# feature importance
importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance)

# plot importance
plt.figure(figsize=(8, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.savefig("plot_importance.png")
plt.show()

# cross validation

scores = cross_val_score( rf,x,y,cv=5,scoring="accuracy")

print("Scores:", scores)
print("Mean Accuracy:", round(scores.mean(), 3))
print("Standard Deviation:", round(scores.std(), 3))