
class AbstractDataLoader():

    def __init__(self, data_path, location_kw):
        self.interim_path = None
        self.data_df = None
        self.data_path = data_path
        self.location_kw = location_kw
        self.load()

    def load(self):
        pass

    def get_interim_path(self):
        return self.interim_path


class AbstractTrajectoryEstimator:
    """ The AbstractTrajectoryEstimator defines the interface for each trajectory estimation algorithm."""

    def __init__(self, imus):
        """
        Initialization of an AbstractTrajectoryEstimator

        Args:
            imus dict[str, IMU]: Dictionary of IMU objects for each sensor location
        """
        self.imus = imus

    def estimate(self, interim_base_path, dataset, subject_num, run_num):
        """
        This method is expected to be implemented by each trajectory estimation algorithm.

        Args:
            interim_base_path (str): Base folder where interim data can be stored
            dataset (str): Folder containing the dataset
            subject_num (int): Subject index
            run_num (int): Run index

        Returns:
            dict[str, DataFrame]: DataFrames containing trajectory information for each foot

        """
        pass


class AbstractEventDetector:
    """
    The AbstractEventDetector defines the interface for any kind of event detection algorithm.
    """

    def __init__(self, imus):
        """
        Initialization of an AbstractEventDetector.

        Args:
            imus (dict[str, IMU]): Dictionary of IMU objects for each sensor location.
        """
        self.imus = imus

    def detect(self):
        """
        This method is expected to be implemented by each event detection algorithm.

        Returns:
            (dict[str, dict]): dictionaries containing gait event information for each foot.

        """
        pass


# class AbstractReferenceLoader():
#
#     def __init__(self, dataset, subject, run):
#         self.load(dataset, subject, run)
#
#     def load(self):
#         pass
#
#     def get_data(self):
#         return self.data
#
# class AbstractEvaluator():
#
#     def __init__(self, trajectory, events, reference_data):
#         self.trajectory = trajectory
#         self.events = events
#         self.reference_data = reference_data
#
#     def evaluate(self):
#         pass