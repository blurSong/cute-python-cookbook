import logging
import cutlass.cute as cute


class CutePrintHandler(logging.Handler):
    terminator = '\n'

    def emit(self, record):
        try:
            msg = self.format(record)
            cute.printf("{}", msg)
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().addHandler(logging.StreamHandler())
