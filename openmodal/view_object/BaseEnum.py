class BaseEnum:
    def __len__(self):
        attrs = vars(self)
        static_vars = {k: v for k, v in attrs.items() if not callable(v) and not k.startswith('__')}
        return len(static_vars)