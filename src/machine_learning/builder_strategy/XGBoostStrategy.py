class XGBoostStrategy(ParamGridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(
                             XGBClassifier() 
                            ,n_estimators = [50, 100,200] 
                            ,max_depth = [5, 10]
                            ,learning_rate = [0.01, 0.1, 0.2]
                            ,subsample = [0.8, 1.0]
                            ,colsample_bytree = [0.8, 1.0]
                            )
            .build()
        )