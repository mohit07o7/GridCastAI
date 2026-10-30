"""Split by original eligible-row position, purge by target availability."""
import pandas as pd

def chronological_split(frame, train_end=None, val_end=None, test_end=None):
    n = len(frame)
    train_end = int(n * .70) if train_end is None else train_end
    val_end = int(n * .85) if val_end is None else val_end
    test_end = n if test_end is None else test_end
    if not 0 < train_end < val_end < test_end <= n:
        raise ValueError('Invalid chronological split boundaries')
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:test_end].copy()
    train = train.loc[train['Target Timestamp'] < validation.Timestamp.min()]
    validation = validation.loc[validation['Target Timestamp'] < test.Timestamp.min()]
    if 'quality_eligible' in frame:
        train = train.loc[train.quality_eligible].copy()
        validation = validation.loc[validation.quality_eligible].copy()
        test = test.loc[test.quality_eligible].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError('Empty split after 24-hour purge')
    assert train['Target Timestamp'].max() < validation.Timestamp.min()
    assert validation['Target Timestamp'].max() < test.Timestamp.min()
    return train, validation, test
