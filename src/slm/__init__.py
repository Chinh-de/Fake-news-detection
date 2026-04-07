# SLM package - lazy imports to avoid pulling in heavy dependencies


def __getattr__(name):
    if name == "FakeNewsDataset":
        from src.slm.dataset import FakeNewsDataset
        return FakeNewsDataset
    if name == "load_data_from_csv":
        from src.slm.dataset import load_data_from_csv
        return load_data_from_csv
    if name == "IntegratedSLM":
        from src.slm.model import IntegratedSLM
        return IntegratedSLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FakeNewsDataset", "load_data_from_csv", "IntegratedSLM"]
