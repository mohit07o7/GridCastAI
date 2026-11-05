"""Runtime guards for the presentation server and headless demo."""
def disabled(*args,**kwargs):
    raise RuntimeError('Training and battery optimization are disabled in the dashboard/demo')


def install_guards():
    import xgboost as xgb
    import common.battery_dispatch as battery
    xgb.XGBRegressor.fit=disabled
    xgb.train=disabled
    xgb.Booster.update=disabled
    battery.milp=disabled
    battery.optimize_dispatch=disabled
