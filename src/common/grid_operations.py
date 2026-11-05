"""Retrospective curve scoring and first-action-only physical replay."""
from dataclasses import asdict,replace
import gzip
import json

import numpy as np
import pandas as pd

from common.battery_dispatch import DispatchOptimizer,execute_hour
from common.curve_features import COMPONENTS,HOUR
from common.grid_evaluation import retrospective_target_flags,attach_target_flags

POLICIES=('no_battery','baseline_battery','forecast_battery')


def attach_evaluation(selected,baseline,source):
    keys=['origin','horizon','target_timestamp']
    for frame in [selected,baseline]:
        if frame.duplicated(keys).any():raise ValueError('Duplicate curve forecast keys')
        if not (frame.target_timestamp-frame.origin).eq(pd.to_timedelta(frame.horizon,unit='h')).all():
            raise ValueError('Shifted curve target')
    renamed=baseline.rename(columns={c:'baseline_'+c for c in baseline if c not in keys})
    result=selected.merge(renamed,on=keys,validate='one_to_one',how='outer',indicator=True)
    if not result._merge.eq('both').all():raise ValueError('Unmatched selected/baseline curves')
    result=result.drop(columns='_merge')
    # This is the first attachment of any future measurements/target audit.
    for c in COMPONENTS:
        result['actual_'+c+'_mw']=source[c+'_mw'].reindex(result.target_timestamp).to_numpy()
    result=attach_target_flags(result,retrospective_target_flags(source))
    result['actual_net_mw']=result.actual_demand_mw-result.actual_wind_mw-result.actual_solar_mw
    result['selected_methods_only']=result[[c+'_status' for c in COMPONENTS]].eq('ok').all(axis=1)
    result['delivered']=~result[[c+'_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
    result['baseline_delivered']=~result[['baseline_'+c+'_status' for c in COMPONENTS]].eq('blocked').any(axis=1)
    result['any_fallback']=result[[c+'_status' for c in COMPONENTS]].eq('fallback').any(axis=1)
    result['score_selected_only']=result.trusted_target & result.selected_methods_only & result.baseline_delivered
    result['score_delivered']=result.trusted_target & result.delivered & result.baseline_delivered
    return result


def score_arrays(actual,predicted):
    if not len(actual):return dict(mae_mw=None,rmse_mw=None,wape_pct=None,mean_signed_error_mw=None)
    delta=np.asarray(predicted)-np.asarray(actual)
    denom=float(np.abs(actual).sum())
    return dict(mae_mw=float(np.abs(delta).mean()),rmse_mw=float(np.sqrt(np.mean(delta**2))),
                wape_pct=float(100*np.abs(delta).sum()/denom) if denom else None,
                mean_signed_error_mw=float(delta.mean()))


def forecast_scores(frame):
    output=[]
    months=frame.target_timestamp.dt.strftime('%Y-%m')
    for h in range(1,25):
        horizon=frame.horizon.eq(h)
        for month in ['all',*sorted(months[horizon].unique())]:
            nominal=horizon & (True if month=='all' else months.eq(month))
            for cohort in ['selected_only','delivered']:
                mask=nominal & frame['score_'+cohort]
                for quantity in [*COMPONENTS,'net']:
                    actual=frame.actual_net_mw if quantity=='net' else frame['actual_'+quantity+'_mw']
                    pred=frame.net_demand_mw if quantity=='net' else frame[quantity+'_mw']
                    base=frame.baseline_net_demand_mw if quantity=='net' else frame['baseline_'+quantity+'_mw']
                    if not np.isfinite(np.c_[actual[mask],pred[mask],base[mask]]).all():
                        raise ValueError('Nonfinite paired forecast scoring values')
                    a=score_arrays(actual[mask],pred[mask]);b=score_arrays(actual[mask],base[mask])
                    improvement=b['mae_mw']-a['mae_mw'] if mask.any() else None
                    output.append(dict(horizon=h,month=month,cohort=cohort,quantity=quantity,
                        nominal_rows=int(nominal.sum()),trusted_targets=int((nominal & frame.trusted_target).sum()),
                        rows=int(mask.sum()),coverage_trusted_pct=float(100*mask.sum()/(nominal & frame.trusted_target).sum()) if (nominal & frame.trusted_target).any() else None,
                        **{'forecast_'+k:v for k,v in a.items()},**{'baseline_'+k:v for k,v in b.items()},
                        mae_improvement_mw=improvement,
                        mae_improvement_pct=100*improvement/b['mae_mw'] if mask.any() and b['mae_mw'] else None))
    return pd.DataFrame(output)


class PlanSink:
    def __init__(self,path):
        self.file=gzip.open(path,'wt',newline='')
        self.buffer=[];self.header=True;self.rows=0
    def add(self,frame):
        self.buffer.append(frame);self.rows+=len(frame)
        if len(self.buffer)>=100:self.flush()
    def flush(self):
        if self.buffer:
            pd.concat(self.buffer,ignore_index=True).to_csv(self.file,index=False,header=self.header,float_format='%.17g')
            self.header=False;self.buffer=[]
    def close(self):self.flush();self.file.close()


def simulate(selected,baseline,source,scenario,sink=None,progress=None):
    """Plans precede target-quality decisions; actuals only reconcile hour one."""
    origins=pd.DatetimeIndex(selected.origin.drop_duplicates().sort_values())
    selected_groups={o:g for o,g in selected.groupby('origin',sort=False)}
    baseline_groups={o:g for o,g in baseline.groupby('origin',sort=False)}
    target_flags=retrospective_target_flags(source)
    optimizer=DispatchOptimizer(scenario)
    no_battery=replace(scenario,battery_capacity_mwh=0.)
    states={p:scenario.initial_soc_mwh for p in POLICIES[1:]}
    pending_reset=True;episode=0;records=[]
    for position,origin in enumerate(origins):
        target=origin+HOUR
        if pending_reset:
            states={p:scenario.initial_soc_mwh for p in POLICIES[1:]}
        # No current target actual/flag has been read at this point.
        plans={}
        for policy,curves in [('baseline_battery',baseline_groups),('forecast_battery',selected_groups)]:
            plan=optimizer.solve(curves[origin],states[policy])
            plans[policy]=plan
            if plan['status']=='optimal' and sink is not None:
                full=plan['hours'].copy()
                full.insert(0,'origin',origin);full.insert(0,'policy',policy);full.insert(0,'scenario',scenario.scenario_id)
                full['minimum_unmet_mwh']=plan['minimum_unmet_mwh']
                sink.add(full)
        actual=source.reindex([target])
        if target in target_flags.index:
            flags=target_flags.loc[target]
            exclusions=[k for k,v in flags.items() if v]
        else:exclusions=['target_missing_observation']
        trusted=not exclusions
        start=bool(pending_reset and trusted)
        if start:episode+=1
        for policy in POLICIES:
            plan=plans.get(policy)
            plan_ok=plan is not None and plan['status']=='optimal'
            p=float(plan['hours'].charge_mw.iloc[0]) if plan_ok else 0.
            q=float(plan['hours'].discharge_mw.iloc[0]) if plan_ok else 0.
            row={'scenario':scenario.scenario_id,'policy':policy,'origin':origin,'target_timestamp':target,
                'trusted_target':trusted,'exclusion_reasons':json.dumps(exclusions),
                'episode_id':episode if trusted else None,'episode_start':start,
                'reset_reason':('initial_or_after_untrusted_actual' if start else None),
                'plan_status':plan['status'] if plan else 'not_applicable',
                'plan_reasons':json.dumps(plan['reasons']) if plan else '[]',
                'plan_predicted_unmet_mwh':plan['predicted_unmet_mwh'] if plan_ok else None,
                'plan_predicted_electricity_cost_inr':plan['predicted_electricity_cost_inr'] if plan_ok else None,
                'plan_predicted_wear_cost_inr':plan['predicted_wear_cost_inr'] if plan_ok else None,
                'planned_charge_mw':p,'planned_discharge_mw':q}
            if trusted:
                values=actual.iloc[0]
                s=no_battery if policy=='no_battery' else scenario
                initial=0. if policy=='no_battery' else states[policy]
                executed=execute_hour(*(float(values[c+'_mw'])*scenario.scale_factor for c in COMPONENTS),
                    p,q,initial,target,s)
                row.update(executed)
                if policy!='no_battery':states[policy]=executed['soc_end_mwh']
            records.append(row)
        pending_reset=not trusted
        if progress and (position+1)%500==0:progress(position+1,len(origins))
    return pd.DataFrame(records)


def operations_scores(frame):
    output=[]
    months=frame.target_timestamp.dt.strftime('%Y-%m')
    for (scenario,policy),group in frame.groupby(['scenario','policy']):
        for month in ['all',*sorted(months.loc[group.index].unique())]:
            g=group if month=='all' else group[months.loc[group.index].eq(month)]
            trusted=g[g.trusted_target]
            energy=float(trusted.demand_mw.sum())
            electricity=float(trusted.electricity_cost_inr.sum());wear=float(trusted.wear_cost_inr.sum())
            output.append(dict(scenario=scenario,policy=policy,month=month,nominal_hours=len(g),
                trusted_hours=len(trusted),excluded_hours=len(g)-len(trusted),
                episode_starts=int(trusted.episode_start.sum()),
                unavailable_plan_hours=int(trusted.plan_status.eq('unavailable').sum()),
                available_plan_pct=100*trusted.plan_status.eq('optimal').mean() if policy!='no_battery' and len(trusted) else None,
                realized_electricity_cost_inr=electricity,realized_wear_cost_inr=wear,
                realized_total_cost_inr=electricity+wear,demand_mwh=energy,
                unserved_mwh=float(trusted.unmet_mw.sum()),
                unserved_hours=int(trusted.unmet_mw.gt(2e-5).sum()),
                demand_served_pct=100*(1-trusted.unmet_mw.sum()/energy) if energy else None,
                peak_grid_import_mw=float(trusted.grid_mw.max()) if len(trusted) else None,
                curtailment_mwh=float(trusted.curtailment_mw.sum()),
                battery_throughput_mwh=float((trusted.charge_mw+trusted.discharge_mw).sum()),
                execution_adjustment_hours=int(trusted.action_adjusted.sum())))
    return pd.DataFrame(output)
