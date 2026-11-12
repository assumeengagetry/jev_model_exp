class Unsupported(Exception):
    pass


class Engine:
    def __init__(self, **options):
        self.options = options
