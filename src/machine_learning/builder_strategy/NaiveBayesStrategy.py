from sklearn.naive_bayes import GaussianNB

class GaussianNBStrategy(ParamGridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(
                             GaussianNB() 
                            ,var_smoothing=[1e-9, 1e-7]
                            )
            .build()
        )