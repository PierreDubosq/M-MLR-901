import black
import grin
import pandas
import jupyterlab
import kneed
import matplotlib.pyplot
import mypy
import numpy
import pandas
import sklearn
import scipy
import seaborn
import snakeviz
import termcolor
import wikipedia
import enum
import datetime
import sys
import math


DATA_PATH = 'data.csv'

class Color(enum.Enum):
  DEFAULT = '\033[0m'
  BLACK = '\033[30m'
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  MAGENTA = '\033[35m'
  CYAN = '\033[36m'
  WHITE = '\033[37m'
  BRIGHT_BLACK = '\033[90m'
  BRIGHT_RED = '\033[91m'
  BRIGHT_GREEN = '\033[92m'
  BRIGHT_YELLOW = '\033[93m'
  BRIGHT_BLUE = '\033[94m'
  BRIGHT_MAGENTA = '\033[95m'
  BRIGHT_CYAN = '\033[96m'
  BRIGHT_WHITE = '\033[97m'



class Logger:

  """
  A class for logging messages with different log levels.

  Args:
    name (str): The name of the logger.
    benchmark (bool, optional): Whether to enable benchmark logging. Defaults to False.

  Methods:
    benchmark(message: str) -> None: Logs a benchmark message if benchmark logging is enabled.
    debug(message: str) -> None: Logs a debug message.
    error(message: str) -> None: Logs an error message.
    info(message: str) -> None: Logs an info message.
    warning(message: str) -> None: Logs a warning message.
  """

  def __init__(self, name: str) -> None:
    self._name: str = name


  def debug(self, message: str) -> None:
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.MAGENTA.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stdout)


  def error(self, message: str) -> None:
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_RED.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stderr)


  def info(self, message: str) -> None:
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_GREEN.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stdout)


  def warning(self, message: str) -> None:
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_YELLOW.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stderr)


def main() -> None:
  logger = Logger('main')

  try:
    Z: pandas.DataFrame = pandas.read_csv(DATA_PATH)[['Year', 'Mean male height (cm)']].groupby('Year').mean().reset_index()
    expected_value: int = Z['Mean male height (cm)'].mean()

    # Choose the number of points to sample
    n = 100

    # Sample 'n' points from the DataFrame Z
    sampled_points = Z.sample(n)

    # Plot the sampled points
    matplotlib.pyplot.scatter(sampled_points['Year'], sampled_points['Mean male height (cm)'], marker='o', color='blue')
    matplotlib.pyplot.xlabel('Year')
    matplotlib.pyplot.ylabel('Mean male height (cm)')
    matplotlib.pyplot.title(f'Sampled Points from DataFrame Z (n={n})')
    matplotlib.pyplot.show()

    # Initialize arrays to store results
    empirical_averages = []
    euclidean_distances = []

    # Sample for increasing values of n
    for i in range(1, n + 1):
        # Extract the first 'i' samples
        subset_samples = sampled_points.head(i)
        
        # Calculate empirical average
        empirical_average = subset_samples.mean()
        
        # Calculate Euclidean distance
        distance = numpy.linalg.norm(empirical_average - expected_value)
        
        # Store results
        empirical_averages.append(empirical_average)
        euclidean_distances.append(distance)

    # Convert lists to DataFrame for easier calculations
    empirical_averages = pandas.DataFrame(empirical_averages)
    euclidean_distances = numpy.array(euclidean_distances)

    # Plot the convergence
    matplotlib.pyplot.plot(range(1, n + 1), euclidean_distances, marker='o', color='blue')
    matplotlib.pyplot.xlabel('Number of Samples (n)')
    matplotlib.pyplot.ylabel('Euclidean Distance')
    matplotlib.pyplot.title('Convergence of Empirical Average to Expected Value')
    matplotlib.pyplot.axhline(0, color='red', linestyle='--', label='Expected Value')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
  except Exception as e:
    logger.error(e)
    raise e


if __name__ == '__main__':
  logger = Logger('system')

  try:
    main()
  except Exception as e:
    logger.error(e)
    sys.exit(1)
