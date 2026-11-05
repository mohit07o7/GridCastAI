"""Illustrative one-hour battery simulation; no market or dispatch claims."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack


@dataclass(frozen=True)
class Scenario:
    scenario_id: str = 'operations_v1'
    scale_factor: float = 100/239406.0305989583
    training_peak_mw: float = 239406.0305989583
    grid_limit_mw: float = 80.
    battery_capacity_mwh: float = 80.
    battery_power_mw: float = 20.
    charge_efficiency: float = float(np.sqrt(.9))
    discharge_efficiency: float = float(np.sqrt(.9))
    min_soc_fraction: float = .1
    max_soc_fraction: float = .9
    initial_soc_fraction: float = .5
    terminal_soc_fraction: float = .5
    wear_cost_inr_per_mwh: float = 250.
    hourly_prices_inr_per_mwh: tuple = (3000.,)*6+(5000.,)*11+(8000.,)*5+(5000.,)*2
    time_step_hours: float = 1.

    def __post_init__(self):
        numeric=[v for k,v in asdict(self).items() if k not in ['scenario_id','hourly_prices_inr_per_mwh']]
        if any(isinstance(v,(bool,complex)) or not np.isfinite(v) for v in numeric):
            raise ValueError('Scenario parameters must be finite real numbers')
        if (self.scale_factor<=0 or self.training_peak_mw<=0 or self.grid_limit_mw<0
                or self.battery_capacity_mwh<0 or self.battery_power_mw<0
                or self.wear_cost_inr_per_mwh<0 or self.time_step_hours!=1
                or not 0<self.charge_efficiency<=1 or not 0<self.discharge_efficiency<=1
                or not 0<=self.min_soc_fraction<=self.initial_soc_fraction<=self.max_soc_fraction<=1
                or not self.min_soc_fraction<=self.terminal_soc_fraction<=self.max_soc_fraction):
            raise ValueError('Invalid physical scenario limits')
        prices=np.asarray(self.hourly_prices_inr_per_mwh)
        if prices.shape!=(24,) or prices.dtype.kind not in 'iuf' or not np.isfinite(prices).all() or (prices<0).any():
            raise ValueError('Require 24 finite nonnegative illustrative prices')

    @property
    def initial_soc_mwh(self):return self.initial_soc_fraction*self.battery_capacity_mwh
    @property
    def minimum_soc_mwh(self):return self.min_soc_fraction*self.battery_capacity_mwh
    @property
    def maximum_soc_mwh(self):return self.max_soc_fraction*self.battery_capacity_mwh
    @property
    def terminal_soc_mwh(self):return self.terminal_soc_fraction*self.battery_capacity_mwh
    @property
    def power(self):return self.battery_power_mw if self.battery_capacity_mwh else 0.


def load_scenario(name,root):
    path=Path(root)/'configs/operations_v1.json' if str(name)=='operations_v1' else Path(name)
    content=json.loads(path.read_text())
    return Scenario(**content['parameters'])


def scenario_variants(base):
    return [base,replace(base,scenario_id='grid_70mw',grid_limit_mw=70.),
            replace(base,scenario_id='grid_100mw',grid_limit_mw=100.),
            replace(base,scenario_id='battery_40mwh',battery_capacity_mwh=40.),
            replace(base,scenario_id='battery_160mwh',battery_capacity_mwh=160.)]


def curve_frame(curve):
    if isinstance(curve,dict):
        rows=[]
        for p in curve['points']:
            row={'horizon':p['horizon'],'target_timestamp':p['target_timestamp']}
            for c in ['demand','wind','solar']:
                row[c+'_mw']=p['components'][c]['forecast_mw']
                if p['components'][c]['status']=='blocked':row[c+'_mw']=np.nan
            rows.append(row)
        frame=pd.DataFrame(rows)
    else:frame=curve.copy()
    frame=frame.sort_values('horizon').reset_index(drop=True)
    if len(frame)!=24 or frame.horizon.tolist()!=list(range(1,25)):
        raise ValueError('Require exactly horizons 1--24')
    times=pd.to_datetime(frame.target_timestamp,errors='raise')
    if (times.isna().any() or times.dt.tz is not None or times.ne(times.dt.floor('h')).any()
            or times.diff().iloc[1:].ne(pd.Timedelta(hours=1)).any()):
        raise ValueError('Require consecutive timezone-naive hourly targets')
    frame['target_timestamp']=times
    if not np.isfinite(frame[['demand_mw','wind_mw','solar_mw']].to_numpy(dtype=float)).all():
        raise ValueError('Incomplete curve: missing/blocked component')
    if frame[['demand_mw','wind_mw','solar_mw']].lt(0).any().any():
        raise ValueError('Negative component forecast')
    return frame


def validate_dispatch(frame,scenario,initial_soc_mwh,tolerance=2e-5):
    """Check exported physical quantities independently of solver matrices."""
    f=frame
    if len(f)!=24:raise ValueError('Expected 24 dispatch hours')
    check=lambda value: bool(np.max(np.asarray(value))<=tolerance)
    if not np.isfinite(f.select_dtypes(include='number').to_numpy()).all():raise ValueError('Nonfinite dispatch')
    nonnegative=['grid_mw','renewable_used_mw','charge_mw','discharge_mw','unmet_mw']
    checks=[check(-f[nonnegative].to_numpy()),check(f.grid_mw-scenario.grid_limit_mw),
        check(f.charge_mw-scenario.power),check(f.discharge_mw-scenario.power),
        check(f.renewable_used_mw-f.wind_mw-f.solar_mw),check(f.unmet_mw-f.demand_mw),
        check(np.minimum(f.charge_mw,f.discharge_mw)),
        check(np.abs(f.grid_mw+f.renewable_used_mw+f.discharge_mw+f.unmet_mw-f.demand_mw-f.charge_mw)),
        check(np.abs(f.soc_end_mwh-f.soc_start_mwh-scenario.charge_efficiency*f.charge_mw+f.discharge_mw/scenario.discharge_efficiency)),
        check(scenario.minimum_soc_mwh-f[['soc_start_mwh','soc_end_mwh']].to_numpy()),
        check(f[['soc_start_mwh','soc_end_mwh']].to_numpy()-scenario.maximum_soc_mwh),
        check(abs(f.soc_start_mwh.iloc[0]-initial_soc_mwh)),
        check(np.abs(f.soc_start_mwh.iloc[1:].to_numpy()-f.soc_end_mwh.iloc[:-1].to_numpy())),
        check(scenario.terminal_soc_mwh-f.soc_end_mwh.iloc[-1])]
    if not all(checks):raise ValueError(f'Dispatch constraint failure: {checks}')
    return True


class DispatchOptimizer:
    """Compile fixed 24-hour MILP structure once per scenario."""
    def __init__(self,scenario):
        self.scenario=scenario
        self.g,self.r,self.c,self.d,self.u,self.e,self.z=[slice(a,b) for a,b in
            [(0,24),(24,48),(48,72),(72,96),(96,120),(120,145),(145,169)]]
        a=lil_matrix((96,169),dtype=float)
        for h in range(24):
            a[h,h]=1;a[h,24+h]=1;a[h,48+h]=-1;a[h,72+h]=1;a[h,96+h]=1
            a[24+h,120+h]=-1;a[24+h,121+h]=1
            a[24+h,48+h]=-scenario.charge_efficiency;a[24+h,72+h]=1/scenario.discharge_efficiency
            a[48+h,48+h]=1;a[48+h,145+h]=-scenario.power
            a[72+h,72+h]=1;a[72+h,145+h]=scenario.power
        self.matrix=a.tocsc()
        reliability=lil_matrix((1,169),dtype=float);reliability[0,self.u]=1
        self.reliability_matrix=reliability.tocsc()
        self.second_matrix=vstack([self.matrix,self.reliability_matrix],format='csc')
        self.first_cost=np.zeros(169);self.first_cost[self.u]=1
        self.integrality=np.zeros(169);self.integrality[self.z]=1

    def solve(self,curve,initial_soc_mwh=None):
        scenario=self.scenario
        initial=scenario.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        if (isinstance(initial,(bool,complex)) or not np.isfinite(initial)
                or initial<scenario.minimum_soc_mwh-1e-7 or initial>scenario.maximum_soc_mwh+1e-7):
            raise ValueError('Initial battery energy lies outside operating bounds')
        try:frame=curve_frame(curve)
        except (ValueError,TypeError,KeyError) as exc:
            return {'status':'unavailable','reasons':[str(exc)],'hours':[]}
        demand=frame.demand_mw.to_numpy(dtype=float)*scenario.scale_factor
        wind=frame.wind_mw.to_numpy(dtype=float)*scenario.scale_factor
        solar=frame.solar_mw.to_numpy(dtype=float)*scenario.scale_factor
        prices=np.asarray(scenario.hourly_prices_inr_per_mwh)[frame.target_timestamp.dt.hour]
        low=np.zeros(169);high=np.full(169,np.inf)
        high[self.g]=scenario.grid_limit_mw;high[self.r]=wind+solar
        high[self.c]=scenario.power;high[self.d]=scenario.power;high[self.u]=demand
        low[self.e]=scenario.minimum_soc_mwh;high[self.e]=scenario.maximum_soc_mwh
        low[120]=high[120]=initial
        low[144]=max(low[144],scenario.terminal_soc_mwh);high[self.z]=1
        lower=np.r_[demand,np.zeros(24),np.full(48,-np.inf)]
        upper=np.r_[demand,np.zeros(24),np.zeros(24),np.full(24,scenario.power)]
        options={'time_limit':10.,'mip_rel_gap':1e-7,'presolve':True}
        try:
            first=milp(self.first_cost,integrality=self.integrality,bounds=Bounds(low,high),
                       constraints=LinearConstraint(self.matrix,lower,upper),options=options)
            if not first.success or first.status!=0:
                return {'status':'unavailable','reasons':[f'reliability_solver:{first.message}'],'hours':[]}
            cost=np.zeros(169);cost[self.g]=prices
            cost[self.c]=scenario.wear_cost_inr_per_mwh;cost[self.d]=scenario.wear_cost_inr_per_mwh
            second=milp(cost,integrality=self.integrality,bounds=Bounds(low,high),
                constraints=LinearConstraint(self.second_matrix,np.r_[lower,-np.inf],np.r_[upper,first.fun+1e-7]),options=options)
            if not second.success or second.status!=0:
                return {'status':'unavailable','reasons':[f'cost_solver:{second.message}'],'hours':[]}
            x=second.x
            result=pd.DataFrame({'horizon':np.arange(1,25),'target_timestamp':frame.target_timestamp,
                'demand_mw':demand,'wind_mw':wind,'solar_mw':solar,'grid_mw':x[self.g],
                'renewable_used_mw':x[self.r],'charge_mw':x[self.c],'discharge_mw':x[self.d],
                'unmet_mw':x[self.u],'soc_start_mwh':x[self.e][:-1],'soc_end_mwh':x[self.e][1:],
                'price_inr_per_mwh':prices})
            result['curtailment_mw']=wind+solar-result.renewable_used_mw
            result['electricity_cost_inr']=result.grid_mw*prices
            result['wear_cost_inr']=(result.charge_mw+result.discharge_mw)*scenario.wear_cost_inr_per_mwh
            validate_dispatch(result,scenario,initial)
            if result.unmet_mw.sum()>first.fun+2e-5:raise ValueError('Reliability optimum was not preserved')
            return {'status':'optimal','reasons':[],'hours':result,
                'minimum_unmet_mwh':float(first.fun),'predicted_unmet_mwh':float(result.unmet_mw.sum()),
                'predicted_electricity_cost_inr':float(result.electricity_cost_inr.sum()),
                'predicted_wear_cost_inr':float(result.wear_cost_inr.sum()),
                'fully_served':bool(result.unmet_mw.sum()<2e-5)}
        except (ValueError,RuntimeError) as exc:
            return {'status':'unavailable','reasons':[f'solver_or_constraint_failure:{exc}'],'hours':[]}


def optimize_dispatch(curve,scenario,initial_soc_mwh=None):
    return DispatchOptimizer(scenario).solve(curve,initial_soc_mwh)


def execute_hour(demand_mw,wind_mw,solar_mw,planned_charge_mw,planned_discharge_mw,
                 initial_soc_mwh,target_timestamp,scenario):
    """Only the just-observed hour reconciles a previously fixed first action."""
    values=np.array([demand_mw,wind_mw,solar_mw,planned_charge_mw,planned_discharge_mw,initial_soc_mwh],dtype=float)
    if not np.isfinite(values).all() or (values < -2e-5).any():raise ValueError('Invalid executed-hour values')
    planned_charge_mw=max(0.,planned_charge_mw)
    planned_discharge_mw=max(0.,planned_discharge_mw)
    if planned_charge_mw>2e-5 and planned_discharge_mw>2e-5:raise ValueError('Simultaneous planned charge/discharge')
    if not scenario.minimum_soc_mwh-2e-5<=initial_soc_mwh<=scenario.maximum_soc_mwh+2e-5:
        raise ValueError('Invalid executed-hour initial charge')
    renewable=wind_mw+solar_mw
    charge=min(planned_charge_mw,scenario.power,
               max(0,(scenario.maximum_soc_mwh-initial_soc_mwh)/scenario.charge_efficiency),
               max(0,renewable+scenario.grid_limit_mw-demand_mw))
    discharge=min(planned_discharge_mw,scenario.power,
                  max(0,(initial_soc_mwh-scenario.minimum_soc_mwh)*scenario.discharge_efficiency),
                  max(0,demand_mw-renewable))
    requirement=max(0,demand_mw+charge-discharge)
    used=min(renewable,requirement)
    grid=min(scenario.grid_limit_mw,max(0,requirement-used))
    unmet=max(0,requirement-used-grid)
    end=initial_soc_mwh+scenario.charge_efficiency*charge-discharge/scenario.discharge_efficiency
    price=scenario.hourly_prices_inr_per_mwh[pd.Timestamp(target_timestamp).hour]
    return {'demand_mw':float(demand_mw),'wind_mw':float(wind_mw),'solar_mw':float(solar_mw),
        'planned_charge_mw':float(planned_charge_mw),'planned_discharge_mw':float(planned_discharge_mw),
        'charge_mw':float(charge),'discharge_mw':float(discharge),'grid_mw':float(grid),
        'renewable_used_mw':float(used),'curtailment_mw':float(renewable-used),'unmet_mw':float(unmet),
        'soc_start_mwh':float(initial_soc_mwh),'soc_end_mwh':float(end),
        'price_inr_per_mwh':float(price),'electricity_cost_inr':float(grid*price),
        'wear_cost_inr':float((charge+discharge)*scenario.wear_cost_inr_per_mwh),
        'action_adjusted':bool(abs(charge-planned_charge_mw)>2e-5 or abs(discharge-planned_discharge_mw)>2e-5)}
