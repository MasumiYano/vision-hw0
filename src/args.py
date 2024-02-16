import sys


def del_arg(index):
    """Deletes an argument from the sys.argv list at the specified index."""
    if index < len(sys.argv):
        del sys.argv[index]


def find_arg(arg):
    """Finds and removes an argument from sys.argv. Returns True if found, False otherwise."""
    if arg in sys.argv:
        del_arg(sys.argv.index(arg))
        return True
    return False


def find_int_arg(arg, default):
    """Finds, removes, and returns an integer argument from sys.argv. Returns default if not found."""
    try:
        index = sys.argv.index(arg)
        value = int(sys.argv[index + 1])
        del_arg(index)
        del_arg(index)  # Remove the value as well
        return value
    except (ValueError, IndexError):
        return default


def find_float_arg(arg, default):
    """Finds, removes, and returns a float argument from sys.argv. Returns default if not found."""
    try:
        index = sys.argv.index(arg)
        value = float(sys.argv[index + 1])
        del_arg(index)
        del_arg(index)  # Remove the value as well
        return value
    except (ValueError, IndexError):
        return default


def find_char_arg(arg, default):
    """Finds, removes, and returns a string argument from sys.argv. Returns default if not found."""
    try:
        index = sys.argv.index(arg)
        value = sys.argv[index + 1]
        del_arg(index)
        del_arg(index)  # Remove the value as well
        return value
    except IndexError:
        return default
