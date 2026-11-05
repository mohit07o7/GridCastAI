"""GridCastAI local presentation; load-only forecasting, optimization deferred."""
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dashboard.service import (ROOT,COMPONENTS,NORMAL_ORIGIN,FALLBACK_ORIGIN,REPORT_DIR,
    signatures,model_paths,report_paths,load_reports,load_history,load_forecaster,forecast,
    forecast_table,attach_actuals,metric_record,audit_slice,csv_bytes,json_bytes)

COLORS={'Demand':'#087f8c','Wind':'#3779bd','Solar':'#d69413','Renewable generation':'#21966c','Net demand':'#7957ad'}


@st.cache_data(show_spinner=False)
def cached_reports(root,signature):return load_reports(Path(root))


@st.cache_data(show_spinner=False)
def cached_history(root,signature):return load_history(Path(root))


@st.cache_resource(show_spinner=False)
def cached_forecaster(root,signature):return load_forecaster(Path(root))


def preset(case,origin):
    stamp=pd.Timestamp(origin)
    st.session_state['issue_date']=stamp.date();st.session_state['issue_hour']=stamp.hour
    st.session_state['case']=case;st.session_state['generate_requested']=True


def line_figure(frame,series,title,ylabel='Power (MW)'):
    figure=go.Figure()
    for name,column in series:
        figure.add_trace(go.Scatter(x=frame.target_timestamp,y=frame[column],mode='lines+markers',
            name=name,connectgaps=False,line=dict(color=COLORS.get(name,'#7b8794'),width=2),marker=dict(size=4)))
    figure.update_layout(template='plotly_white',height=330,title=title,hovermode='x unified',
        margin=dict(l=15,r=15,t=50,b=20),legend=dict(orientation='h',y=-.25),
        xaxis_title='Target timestamp · source clock',yaxis_title=ylabel)
    return figure


def number(value,suffix=''):
    return 'Unavailable' if pd.isna(value) else f'{value:,.2f}{suffix}'


def main():
    st.set_page_config(page_title='GridCastAI · Forecast desk',page_icon='⚡',layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1450px;padding-top:4rem;padding-bottom:3rem;}
    h1 {letter-spacing:-1.5px;} [data-testid="stMetric"] {background:#f3f8f7;padding:16px;border-radius:10px;}
    [data-testid="stMetricValue"] {font-size:1.25rem;}
    [data-testid="stSidebar"] {background:#f4f7f8;} .eyebrow {color:#087f8c;font-weight:700;letter-spacing:2px;font-size:12px;}
    </style>''',unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">GRIDCASTAI / LOCAL DEMONSTRATION</div>',unsafe_allow_html=True)
    st.title('Forecast desk')
    st.caption('24-hour forecasts · historical evidence · quality audit. Historical runs are retrospective simulations.')
    report_signature=signatures(report_paths(ROOT));model_signature=signatures(model_paths(ROOT))
    history_signature=signatures([ROOT/'data/processed/full_hourly_stacked.csv',ROOT/REPORT_DIR/'summary.json'])
    reports=None;history=None;model=None;report_error=None;forecast_error=None
    try:reports=cached_reports(str(ROOT),report_signature)
    except Exception as exc:report_error=str(exc)
    try:
        history=cached_history(str(ROOT),history_signature)
        model=cached_forecaster(str(ROOT),(model_signature,report_signature))
    except Exception as exc:forecast_error=str(exc)
    st.sidebar.header('Issue a forecast')
    if 'issue_date' not in st.session_state:st.session_state['issue_date']=pd.Timestamp(NORMAL_ORIGIN).date()
    if 'issue_hour' not in st.session_state:st.session_state['issue_hour']=23
    if 'case' not in st.session_state:st.session_state['case']='normal'
    st.sidebar.date_input('Origin date',key='issue_date')
    st.sidebar.selectbox('Origin hour',list(range(24)),key='issue_hour',format_func=lambda h:f'{h:02d}:00')
    st.sidebar.caption('Source-clock hours. The observation at this hour is assumed available.')
    generate=st.sidebar.button('Generate forecast',type='primary',width='stretch')
    st.sidebar.markdown('**Try a complete example**')
    st.sidebar.button('Normal forecast',on_click=preset,args=('normal',NORMAL_ORIGIN),width='stretch')
    st.sidebar.button('Historical fallback',on_click=preset,args=('fallback',FALLBACK_ORIGIN),width='stretch')
    st.sidebar.button('Synthetic blocked wind',on_click=preset,args=('blocked',NORMAL_ORIGIN),width='stretch')
    if history is not None:
        st.sidebar.caption(f'Available observations: {history.timestamp.min():%d %b %Y} – {history.timestamp.max():%d %b %Y}.')
    st.sidebar.divider();st.sidebar.markdown('**Optimization: Deferred**')
    st.sidebar.caption('No battery schedule, dispatch action or savings estimate is generated.')
    pending=st.session_state.pop('generate_requested',False)
    signature=(report_signature,model_signature,history_signature)
    if generate or pending or 'forecast_attempted' not in st.session_state:
        st.session_state['forecast_attempted']=True
        if forecast_error is None:
            origin=pd.Timestamp(st.session_state.issue_date)+pd.Timedelta(hours=st.session_state.issue_hour)
            with st.spinner('Loading the saved forecast methods…'):
                try:
                    result=forecast(model,history,origin,st.session_state['case'])
                    st.session_state['result']={'forecast':result,'case':st.session_state['case'],'signature':signature}
                    st.session_state.pop('generation_error',None)
                except Exception as exc:st.session_state['generation_error']=str(exc)
    if report_error:st.error(f'Historical results unavailable: {report_error}')
    if forecast_error:st.warning(f'New forecasts unavailable: {forecast_error}. Historical reports remain accessible when valid.')
    if st.session_state.get('generation_error'):st.error('Forecast request unavailable: '+st.session_state['generation_error'])
    tabs=st.tabs(['24-hour forecast','Historical accuracy','Quality and exclusions','Optimization status'])
    with tabs[0]:
        saved=st.session_state.get('result')
        if saved:
            result=saved['forecast'];table=forecast_table(result)
            st.subheader(f'Issued from {result["origin"]}')
            st.caption(f'Targets {table.target_timestamp.iloc[0]} through {table.target_timestamp.iloc[-1]} · 24 hourly targets · MW')
            if saved['signature']!=signature:st.warning('Artifacts changed after this result was generated. Generate again to use the current files.')
            if saved['case']=='blocked':st.warning('Synthetic demonstration: wind is negative in a temporary input fixture. Source measurements are untouched.')
            if result['status']=='ok':st.success('Complete curve · all selected methods available')
            elif result['complete_curve']:st.warning('Complete curve · clearly labelled fallback methods are included')
            else:st.warning('Incomplete curve · blocked components remain missing; gaps are not filled with zero')
            status=table[[c+'_status' for c in COMPONENTS]]
            cards=st.columns(4)
            cards[0].metric('Peak demand (MW)',number(table.demand_mw.max()))
            cards[1].metric('Peak renewables (MW)',number(table.renewable_mw.max()))
            cards[2].metric('Delivered target hours',f'{result["delivered_target_hours"]} / 24')
            cards[3].metric('Hours with fallback',str(int(status.eq('fallback').any(axis=1).sum())))
            show_actual=st.checkbox('Overlay reported historical actuals',value=False,disabled=saved['case']=='blocked',key='show_actual')
            displayed=attach_actuals(table,history) if history is not None and saved['case']!='blocked' else table
            fig=line_figure(displayed,[('Demand','demand_mw'),('Wind','wind_mw'),('Solar','solar_mw')],'Aligned component forecasts')
            if show_actual and 'actual_available' in displayed:
                for c in COMPONENTS:
                    fig.add_trace(go.Scatter(x=displayed.target_timestamp,y=displayed['actual_'+c+'_mw'],name=c.title()+' actual',connectgaps=False,line=dict(color=COLORS[c.title()],dash='dot')))
                st.caption('Reported actuals are a retrospective overlay; they did not enter the forecast. Inspect exclusions separately.')
            st.plotly_chart(fig,width='stretch',key='component_chart')
            if 'actual_available' not in displayed or not displayed.actual_available.any():st.info('Actual unavailable for these future target hours.')
            elif show_actual:st.caption(f'Reported actuals available for {int(displayed.actual_available.sum())}/24 target hours.')
            a,b=st.columns(2)
            with a:st.plotly_chart(line_figure(table,[('Renewable generation','renewable_mw'),('Net demand','net_demand_mw')],'Renewables and signed net demand'),width='stretch',key='net_chart')
            with b:st.plotly_chart(line_figure(table,[('Renewable contribution','renewable_share_pct')],'Renewable contribution','Share of forecast demand (%)'),width='stretch',key='share_chart')
            st.caption('Net demand = demand − wind − solar. A negative value is surplus. Renewable share is undefined when demand is nonpositive; values above 100% remain visible.')
            columns=['horizon','target_timestamp','demand_mw','wind_mw','solar_mw','renewable_mw','net_demand_mw','renewable_share_pct',*[c+'_status' for c in COMPONENTS]]
            st.dataframe(table[columns],hide_index=True,width='stretch')
            one,two=st.columns(2)
            one.download_button('Download 24-hour CSV',csv_bytes(table),'gridcast_forecast.csv','text/csv',on_click='ignore')
            two.download_button('Download forecast JSON',json_bytes(result),'gridcast_forecast.json','application/json',on_click='ignore')
            with st.expander('About this result'):
                st.write('The issue time is the latest observation allowed into these forecasts. Each horizon uses its own saved direct method. No model is retrained here.')
                st.write('Fallback candidates are target-aligned previous day, previous week and last observation, ordered by saved validation performance. Target quality remains unknown.')
                st.json({'model':result['model_id'],'manifest_sha256':result['manifest_sha256'],'inference_policy':result['inference_quality_policy'],'target_quality':result['target_quality'],'timestamp_convention':result['timestamp_convention']})
            with st.expander('Component methods and input-quality reasons'):
                st.dataframe(table[['horizon','target_timestamp',*[c+'_'+f for c in COMPONENTS for f in ['method','status','reasons','unavailable_checks']]]],hide_index=True,width='stretch')
            if show_actual and 'actual_available' in displayed:
                with st.expander('Reported actuals and renewable-share difference'):
                    st.caption('Renewable-share error is forecast minus reported actual, in percentage points (pp).')
                    st.dataframe(displayed[['target_timestamp','actual_available','actual_demand_mw','actual_wind_mw','actual_solar_mw','actual_renewable_share_pct','renewable_share_error_pp']],hide_index=True)
        else:st.info('Choose an observed origin and generate a forecast. Historical reports are available in the next views.')
    with tabs[1]:
        st.subheader('Historical accuracy')
        st.caption('Retrospective evaluation · model and baseline use identical scored rows within each comparison. These periods have informed development.')
        if reports is not None:
            controls=st.columns(4)
            quantity=controls[0].selectbox('Quantity',['net','demand','wind','solar'],format_func=lambda q:'Net demand' if q=='net' else q.title(),key='quantity')
            horizon=controls[1].selectbox('Forecast horizon',list(range(1,25)),index=23,format_func=lambda h:f'{h} hours',key='horizon')
            cohort=controls[2].selectbox('Scoring cohort',['delivered','selected_only'],format_func=lambda c:'Delivered (includes fallbacks)' if c=='delivered' else 'Selected methods only',key='cohort')
            month=controls[3].selectbox('Target month',['all',*sorted(set(reports.metrics.month)-{'all'})],format_func=lambda m:'All months' if m=='all' else m,key='month')
            record=metric_record(reports,quantity,horizon,cohort,month)
            audited,counts,reasons=audit_slice(reports,horizon,month,cohort)
            c=st.columns(4)
            c[0].metric('Forecast MAE',number(record.forecast_mae_mw,' MW'))
            c[1].metric('Baseline MAE',number(record.baseline_mae_mw,' MW'))
            c[2].metric('MAE reduction',number(record.mae_improvement_pct,'%'))
            c[3].metric('Trusted-target coverage',number(record.coverage_trusted_pct,'%'))
            st.caption(f"Scored origin–horizon forecast records: {counts['scored_records']:,} / {counts['trusted_records']:,} trusted. Unique scored target hours: {counts['scored_unique_target_hours']:,}. Nominal records: {counts['forecast_records']:,}.")
            details=pd.DataFrame({'Metric':['MAE (MW)','RMSE (MW)','WAPE (%)','Mean signed error (MW)'],
                'Forecast':[record['forecast_'+m] for m in ['mae_mw','rmse_mw','wape_pct','mean_signed_error_mw']],
                'Baseline':[record['baseline_'+m] for m in ['mae_mw','rmse_mw','wape_pct','mean_signed_error_mw']]})
            st.dataframe(details,hide_index=True,width='stretch')
            monthly=reports.metrics.query('quantity == @quantity and horizon == @horizon and cohort == @cohort and month != "all"').sort_values('month')
            f=go.Figure()
            for name,column,color in [('Forecast','forecast_mae_mw','#087f8c'),('Baseline','baseline_mae_mw','#8d99a6')]:f.add_trace(go.Bar(x=monthly.month,y=monthly[column],name=name,marker_color=color))
            f.update_layout(title='Monthly paired MAE',template='plotly_white',yaxis_title='MAE (MW)',xaxis_title='Target month',height=330)
            st.plotly_chart(f,width='stretch',key='monthly_chart')
            st.caption('Signed error = forecast − actual. Undefined ratios appear as missing, never zero. Monthly scores use each month’s own identical paired mask.')
            st.download_button('Download selected accuracy row',csv_bytes(record.to_frame().T),'gridcast_accuracy.csv','text/csv',on_click='ignore')
    with tabs[2]:
        st.subheader('Quality and exclusions')
        st.markdown('**At issue time: observed inputs**')
        saved=st.session_state.get('result')
        if saved:
            t=forecast_table(saved['forecast'])
            st.caption(f"Input checks for the displayed forecast issued from {saved['forecast']['origin']}. They do not establish future target quality.")
            st.dataframe(t[['horizon',*[c+'_'+f for c in COMPONENTS for f in ['status','reasons','unavailable_checks']]]],hide_index=True,width='stretch')
        st.markdown('**After observation: retrospective target audit**')
        st.caption('This separate audit determines historical scoring eligibility. It never changes the earlier prediction or fallback decision. Filters follow the Historical accuracy view.')
        if reports is not None:
            audited,counts,reasons=audit_slice(reports,st.session_state.get('horizon',24),st.session_state.get('month','all'),st.session_state.get('cohort','delivered'))
            cols=st.columns(3)
            cols[0].metric('Excluded target records',str(counts['excluded_target_records']))
            cols[1].metric('Trusted but unscored records',str(counts['trusted_unscored_records']))
            cols[2].metric('Records using fallback',str(counts['fallback_records']))
            st.dataframe(reasons,hide_index=True,width='stretch')
            st.caption('Reason counts can overlap. Their sum is not the number of excluded records.')
            excluded=audited[~audited.trusted_target][['origin','horizon','target_timestamp','target_exclusion_reasons']]
            st.dataframe(excluded.head(200),hide_index=True,width='stretch')
            st.download_button('Download all filtered exclusions',csv_bytes(excluded),'gridcast_exclusions.csv','text/csv',on_click='ignore')
            with st.expander('Evaluation provenance'):
                st.json({'verification':reports.verification['status'],'origins':reports.summary['nominal_origins'],'forecast_records':reports.summary['forecast_exposures'],'unique_target_hours_all_horizons':reports.rows.target_timestamp.nunique(),'target_policy':reports.summary['target_policy'],'source':reports.summary['source'],'model_manifest_sha256':reports.summary['model_manifest_sha256']})
    with tabs[3]:
        st.subheader('Optimization: Deferred')
        st.info('Battery optimization is set aside. This dashboard does not run a battery solver or display battery schedules, dispatch decisions or simulated savings.')
        st.write('Signed net demand shows the forecast residual supply requirement. It is a calculation, not an optimized dispatch plan.')


if __name__=='__main__':main()
