import pyqtgraph as pg
import pyqtgraph.multiprocess as mp

from src import log

# TODO - Flesh this out further.
class Visualization:
    def __init__(self):
        self.__app__ = pg.mkQApp()

        # Create remote process with a plot window
        self.__proc__ = mp.QtProcess()
        self.__rpg__ = self.__proc__._import('pyqtgraph')
        self.__plotwin__ = self.__rpg__.plot()
        self.__curve__ = self.__plotwin__.plot(pen='y')

        self.__data__ = self.__proc__.transfer([])

        self.__max__ = 10_000
        self.__sliding__ = 2_500

    def add_data(self, new_data) -> bool:
        if len(new_data) == 0:
            return False
        # create an empty list in the remote process
        # Send new data to the remote process and plot it
        # We use the special argument _callSync='off' because we do
        # not want to wait for a return value.

        try:
            self.__data__.extend(new_data, _callSync='off')

            if len(self.__data__) > self.__max__:
                self.__data__ = self.__proc__.transfer(self.__data__[-self.__sliding__:])

            self.__curve__.setData(y=self.__data__, _callSync='off')

            return True
        except Exception as e:
            log.critical(f"ERROR WITH VISUALIZER: {e}")
            return False