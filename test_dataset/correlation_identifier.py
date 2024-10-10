from ng.DataSynthesizer.DataDescriber import DataDescriber
import warnings
import time
warnings.filterwarnings("ignore")

class CorrIdentifier:

    def __init__(self, dataset, cat_attrs={}, bn_degree=0):
        self.mode = 'correlated_attribute_mode'
        self.input_data = dataset

        # An attribute is categorical if its domain size is less than this threshold.
        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
        self.threshold_value = 20

        # specify categorical attributes
        self.categorical_attributes = cat_attrs

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        self.degree_of_bayesian_network = bn_degree

        self.describer = DataDescriber(category_threshold=self.threshold_value)

    def get_bayesian_network(self, epsilon=1):
        self.describer.describe_dataset_in_correlated_attribute_mode(dataset_file=self.input_data,
                                                                     epsilon=epsilon,
                                                                     k=self.degree_of_bayesian_network,
                                                                     attribute_to_is_categorical=self.categorical_attributes)

        return self.describer.bayesian_network





