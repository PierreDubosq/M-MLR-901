import matplotlib.pyplot
import numpy
import pandas
import enum
import datetime
import sys


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
    """
    Prints a debug message with the current date and time, the name of the object, and the message.
    
    Args:
      message (str): The debug message to be printed.
      
    Returns:
      None
    """
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.MAGENTA.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stdout)


  def error(self, message: str) -> None:
    """
    Prints an error message with the current date and time, the name of the object, and the error message.

    Args:
      message (str): The error message to be printed.

    Returns:
      None
    """
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_RED.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stderr)


  def info(self, message: str) -> None:
    """
    Prints the current date and time, along with the name of the object and the provided message.

    Args:
      message (str): The message to be printed.

    Returns:
      None
    """
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_GREEN.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stdout)


  def warning(self, message: str) -> None:
    """
    Prints a warning message with the current date and time, the name of the object, and the message.

    Args:
      message (str): The warning message.

    Returns:
      None
    """
    date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{Color.BRIGHT_BLUE.value}{date} {Color.BRIGHT_YELLOW.value}[{self._name}] {Color.BRIGHT_YELLOW.value}{message}{Color.DEFAULT.value}', flush=True, file=sys.stderr)


def help():
  """
  Display the help menu with usage and available commands.
  """
  logger = Logger('help')

  try:
    logger.info(f'{"*"*10} HELP {"*"*10}\n')
    logger.info(f'{" "*2}Usage')
    logger.info(f'{" "*4}python main.py [command]\n')
    logger.info(f'{" "*2}Commands')
    logger.info(f'{" "*4}Help\tDisplay this help menu')
    logger.info(f'{" "*4}1   \tExecute part 1')
    logger.info(f'{" "*4}2   \tExecute part 2')
    logger.info(f'{" "*4}3   \tExecute part 3')
  except Exception as e:
    logger.error(e)
    raise e

def generate_random_variable(num_points: int):
  """
  Generate a random variable with given number of points.

  Parameters:
  - num_points (int): The number of points to generate.

  Returns:
  - Z (numpy.ndarray): The generated random variable as a 2D array.
  """
  logger = Logger('generate_random_variable')

  try:
    X = numpy.random.normal(loc=30, scale=10, size=num_points)
    Y = numpy.random.uniform(low=0, high=100, size=num_points)
    Z = numpy.column_stack((X, Y))
    return Z
  except Exception as e:
    logger.error(e)
    raise e

def compute_expected_value(Z):
  """
  Compute the expected value of a given array.

  Parameters:
  Z (numpy.ndarray): The input array.

  Returns:
  numpy.ndarray: The computed expected value.

  Raises:
  Exception: If an error occurs during computation.
  """
  logger = Logger('compute_expected_value')

  try:
    expected_value = 0
    for i in range(0, len(Z)):
      expected_value += Z[i][0] * Z[i][1]
    return expected_value
  except Exception as e:
    logger.error(e)
    raise e

def sample_and_plot(num_points: int):
  """
  Generate random variable samples and plot them.

  Args:
    num_points (int): The number of points to sample.

  Raises:
    Exception: If an error occurs during the sampling or plotting process.

  Returns:
    None
  """
  logger = Logger('sample_and_plot')

  try:
    sampled_points = generate_random_variable(num_points)
    matplotlib.pyplot.scatter(sampled_points[:, 0], sampled_points[:, 1], alpha=0.5)
    matplotlib.pyplot.title(f'Sampled Points from Z ({num_points} points)')
    matplotlib.pyplot.xlabel('X')
    matplotlib.pyplot.ylabel('Y')
    matplotlib.pyplot.show()
  except Exception as e:
    logger.error(e)
    raise e

def compute_and_plot_convergence(Z, max_n):
  """
  Computes and plots the convergence of the empirical average to the expected value.

  Parameters:
  Z (numpy.ndarray): The input data.
  max_n (int): The maximum number of samples.

  Raises:
  Exception: If an error occurs during the computation or plotting.

  Returns:
  None
  """
  logger = Logger('compute_and_plot_convergence')

  try:
    expected_value = compute_expected_value(Z)
    distances = []

    for n in range(1, max_n + 1):
      sampled_points = generate_random_variable(n)
      empirical_average = numpy.mean(sampled_points, axis=0)
      distance = numpy.linalg.norm(empirical_average - expected_value)
      distances.append(distance)

    matplotlib.pyplot.plot(range(1, max_n + 1), distances)
    matplotlib.pyplot.title('Convergence of Empirical Average to Expected Value')
    matplotlib.pyplot.xlabel('Number of Samples (n)')
    matplotlib.pyplot.ylabel('Euclidean Distance')
    matplotlib.pyplot.show()
  except Exception as e:
    logger.error(e)
    raise e


def main(argv: list[str]) -> None:
  """
  Main function that serves as the entry point of the program.

  Args:
    argv (list[str]): Command-line arguments passed to the program.

  Returns:
    None
  """
  logger = Logger('main')

  try:
    if len(argv) != 2:
      help()
      sys.exit(1)

    num_points: int = 1000
    Z = generate_random_variable(num_points)

    if argv[1] == '1':
      logger.info(f'{"*"*10} Part 1 {"*"*10}')
      expected_value = compute_expected_value(Z)
      logger.info(f'Expected Value of Z: {expected_value}')

    elif argv[1] == '2':
      logger.info(f'{"*"*10} Part 2 {"*"*10}')
      sample_and_plot(num_points)

    elif argv[1] == '3':
      logger.info(f'{"*"*10} Part 3 {"*"*10}')
      max_n = num_points * 10
      compute_and_plot_convergence(Z, max_n)
  except Exception as e:
    logger.error(e)
    raise e


if __name__ == '__main__':
  logger = Logger('system')

  try:
    main(sys.argv)
  except Exception as e:
    logger.error(e)
    sys.exit(1)
