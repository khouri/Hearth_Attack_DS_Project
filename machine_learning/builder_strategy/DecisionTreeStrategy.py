from .ParamGridStrategy import ParamGridStrategy
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeStrategy(ParamGridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(
                             DecisionTreeClassifier() ,
                             criterion= ['gini', 'entropy'],
                             max_depth= [None, 3, 5, 10, 20],
                             min_samples_split= [2, 5, 10],
                             min_samples_leaf= [1, 2, 4],
                             max_features= ['sqrt', 'log2', None],
                             splitter= ['best', 'random'],
                             ccp_alpha= [0.0, 0.01, 0.1]  # Poda de complexidade
                           )
            .build()
        )
